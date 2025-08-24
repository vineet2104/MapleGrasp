import time
import functools
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
import numpy as np

from loguru import logger
from tqdm import tqdm

# If your code uses these utilities:
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU)
from utils.grasp_eval import (detect_grasps, calculate_jacquard_index)

################################################################################
# TRAIN LOOP
################################################################################

def train_one_epoch(epoch,
                    model,
                    diffusion,
                    schedule_sampler,
                    train_loader,
                    optimizer,
                    args):
    """
    Train LGDM for one epoch using:
      1) Diffusion loss
      2) Grasp loss (pos/cos/sin/width)
      3) Instance mask loss (smooth L1)
    Summed into final_loss = (diffusion_loss + grasp_loss + mask_loss).

    The function uses progress meters to track IoU, etc. 
    Returns a dictionary of average stats, e.g.:
      {
        'loss': float, 
        'losses': { 'mask_loss':..., 'p_loss':..., ...},
        'iou': float,
        'prec@50': float
      }
    """

    # ------------------------------------------------------------------------
    # Setup Meters
    # ------------------------------------------------------------------------
    batch_time  = AverageMeter('Batch', ':2.2f')
    data_time   = AverageMeter('Data',  ':2.2f')
    lr_meter    = AverageMeter('LR',    ':1.6f')
    loss_meter  = AverageMeter('Loss',  ':2.4f')

    # Sub-losses
    mask_loss_meter  = AverageMeter('MaskLoss', ':2.4f')
    p_loss_meter     = AverageMeter('Loss_pos', ':2.4f')
    cos_loss_meter   = AverageMeter('Loss_cos', ':2.4f')
    sin_loss_meter   = AverageMeter('Loss_sin', ':2.4f')
    width_loss_meter = AverageMeter('Loss_wid', ':2.4f')

    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter  = AverageMeter('Prec@50', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr_meter, loss_meter,
            mask_loss_meter, p_loss_meter, cos_loss_meter, sin_loss_meter, width_loss_meter,
            iou_meter, pr_meter
        ],
        prefix=f"Training Epoch {epoch}: "
    )

    model.train()
    device = torch.device(f'cuda:{args.gpu}') if hasattr(args, 'gpu') else torch.device('cuda')
    scaler = amp.GradScaler()  # or pass in a global scaler if you prefer

    end = time.time()

    for i, data in enumerate(train_loader):
        # --------------------------------------------------------------------
        # 1) Data Prep
        # --------------------------------------------------------------------
        image     = data["img"].to(device, non_blocking=True)
        ins_mask  = data["mask"].to(device, non_blocking=True).unsqueeze(1)          # instance mask GT
        qua_mask  = data["grasp_masks"]["qua"].to(device, non_blocking=True).unsqueeze(1)
        sin_mask  = data["grasp_masks"]["sin"].to(device, non_blocking=True).unsqueeze(1)
        cos_mask  = data["grasp_masks"]["cos"].to(device, non_blocking=True).unsqueeze(1)
        wid_mask  = data["grasp_masks"]["wid"].to(device, non_blocking=True).unsqueeze(1)

        if "sentence" in data:
            query = data["sentence"]  # list[str]
        else:
            query = None

        data_time.update(time.time() - end)
        bs = image.shape[0]

        # --------------------------------------------------------------------
        # 2) Sample random timesteps t from schedule_sampler
        # --------------------------------------------------------------------
        t, weights = schedule_sampler.sample(bs, device)
        idx = torch.zeros(bs, dtype=torch.float, device=device)
        alpha = 0.4  # could vary by epoch or iteration

        # --------------------------------------------------------------------
        # 3) Forward + Loss
        #    -> diffusion loss, grasp loss, instance mask loss
        # --------------------------------------------------------------------
        with amp.autocast(enabled=True):
            # a) Diffusion forward => net(...) sets final predictions inside net
            compute_losses = functools.partial(
                diffusion.training_losses,
                model,
                qua_mask,   # e.g. treat ins_mask as the "pos_gt" if that is how you do it
                image,
                t,
                query,
                alpha,
                idx
            )
            losses_out = compute_losses()
            diffusion_loss = (losses_out["loss"] * weights).mean()  # typical usage

            mask_pred,qua_pred, cos_pred, sin_pred, width_pred = model.module.mask_output_str,model.module.pos_output_str, model.module.cos_output_str, model.module.sin_output_str, model.module.width_output_str

            yc = [ins_mask,qua_mask,cos_mask,sin_mask,wid_mask]

            lossd = model.module.compute_loss(yc,mask_pred,qua_pred,cos_pred,sin_pred,width_pred)

            grasp_loss = lossd['loss']

            # final_loss
            final_loss = diffusion_loss + grasp_loss

        # 4) Backprop
        optimizer.zero_grad()
        scaler.scale(final_loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # 5) Compute IoU vs. instance mask
        # If your net sets e.g. net.pos_output_str or we have mask_pred:
        # We'll assume the "mask_pred" is the final instance mask. 
        # If it's not, adapt accordingly (e.g. if your model uses pos_pred as instance mask).
        if mask_pred is not None:
            ins_mask_pred = mask_pred
        else:
            # fallback if your net uses 'pos' or something as instance mask
            ins_mask_pred = lossd["pred"]["mask"]

        iou_val, pr5_val = trainMetricGPU(ins_mask_pred, ins_mask, threshold=0.35)

        # 6) DDP reduce
        dist.all_reduce(final_loss.detach())
        dist.all_reduce(iou_val)
        dist.all_reduce(pr5_val)

        final_loss_val = final_loss / dist.get_world_size()
        iou_val = iou_val / dist.get_world_size()
        pr5_val = pr5_val / dist.get_world_size()

        # 7) Update meters
        loss_meter.update(final_loss_val.item(), bs)
        

        # Sub-losses from net.compute_loss => p_loss, cos_loss, sin_loss, width_loss
        mask_loss_meter.update(lossd["losses"]["mask_loss"].item(), bs)
        p_loss_meter.update(lossd["losses"]["p_loss"].item(), bs)
        cos_loss_meter.update(lossd["losses"]["cos_loss"].item(), bs)
        sin_loss_meter.update(lossd["losses"]["sin_loss"].item(), bs)
        width_loss_meter.update(lossd["losses"]["width_loss"].item(), bs)

        iou_meter.update(iou_val.item(), bs)
        pr_meter.update(pr5_val.item(), bs)
        lr_meter.update(optimizer.param_groups[0]["lr"])

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)

    # # Gather final stats
    # train_results = {
    #     'loss': loss_meter.avg,
    #     'losses': {
    #         'mask_loss': mask_loss_meter.avg,
    #         'p_loss': p_loss_meter.avg,
    #         'cos_loss': cos_loss_meter.avg,
    #         'sin_loss': sin_loss_meter.avg,
    #         'width_loss': width_loss_meter.avg
    #     },
    #     'iou': iou_meter.avg,
    #     'prec@50': pr_meter.avg
    # }

    # return train_results


################################################################################
# VALIDATION LOOP
################################################################################

@torch.no_grad()
def validate_one_epoch(model,
                       diffusion,
                       schedule_sampler,
                       device,
                       val_loader,
                       epoch,
                       args):
    """
    Validation function combining:
      1) The same LGDM forward pass from train_one_epoch (calls diffusion.training_losses)
      2) The instance/grasp metric logic from crog_engine.py's validate_with_grasp
         (including inverse warping, IoU, top-K Jacquard).

    Returns:
        iou (float): Mean IoU across validation dataset
        prec (dict): { "Pr@50":X, "Pr@60":Y, ... }
        j_index (list): [j1, j5] for top-1 and top-5 Jacquard index
    """

    # For inverse warping to original size (like crog_engine)
    def inverse(img, mat, w, h):
        """Use OpenCV warpAffine to invert back to (w,h)."""
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderValue=0.)
        return inv_img

    model.eval()
    time.sleep(2)

    # Track IoUs
    iou_list = []
    # For Jacquard
    num_grasps = [1, 5]
    num_correct_grasps = [0, 0]
    num_total_grasps   = [0, 0]

    pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}")

    for data_batch in pbar:
        # --------------------------------------------------------------------
        # 1) Fetch data from batch (matching crog_engine)
        # --------------------------------------------------------------------
        image     = data_batch["img"].to(device, non_blocking=True)
        ins_mask  = data_batch["mask"].to(device, non_blocking=True).unsqueeze(1)
        qua_mask  = data_batch["grasp_masks"]["qua"].to(device, non_blocking=True).unsqueeze(1)
        sin_mask  = data_batch["grasp_masks"]["sin"].to(device, non_blocking=True).unsqueeze(1)
        cos_mask  = data_batch["grasp_masks"]["cos"].to(device, non_blocking=True).unsqueeze(1)
        wid_mask  = data_batch["grasp_masks"]["wid"].to(device, non_blocking=True).unsqueeze(1)

        # Depending on your code, text can be a list[str] or a tensor
        # e.g. if (CLIPSEG or LGRCONVNET) => list[str]
        if ("CLIPSEG" in args.exp_name or "LGRCONVNET" in args.exp_name):
            text = data_batch['sentence']
        else:
            # a tensor
            text = data_batch["word_vec"].to(device, non_blocking=True)

        # Additional items
        inverse_matrix = data_batch["inverse"]
        ori_sizes      = data_batch["ori_size"]
        grasp_targets  = data_batch["grasps"]

        # If you have an 'infer_mask':
        if args.use_max_pool:
            infer_mask = data_batch['infer_mask'].to(device, non_blocking=True).unsqueeze(1)
        else:
            infer_mask = None

        bs = image.shape[0]
        alpha = 0.4
        idx = torch.ones(bs, dtype=torch.float, device=device)

        # --------------------------------------------------------------------
        # 2) Diffusion forward pass (mirroring train_one_epoch)
        #    i.e. partial call => diffusion.training_losses => sets model.*output_str
        # --------------------------------------------------------------------
        t, weights = schedule_sampler.sample(bs, device)
        with amp.autocast(enabled=False):
            compute_losses = functools.partial(
                diffusion.training_losses,
                model,
                qua_mask,  # same as in train function
                image,
                t,
                text,      # query
                alpha,
                idx
            )
            losses_out = compute_losses()  # no backward => pure eval

        # After the forward pass, the model's final predictions are stored in:
        #   model.mask_output_str, model.pos_output_str, model.cos_output_str,
        #   model.sin_output_str, model.width_output_str
        if hasattr(model, 'module'):  # DDP
            mask_pred = model.module.mask_output_str
            qua_pred  = model.module.pos_output_str
            cos_pred  = model.module.cos_output_str
            sin_pred  = model.module.sin_output_str
            width_pred= model.module.width_output_str
        else:
            mask_pred = model.mask_output_str
            qua_pred  = model.pos_output_str
            cos_pred  = model.cos_output_str
            sin_pred  = model.sin_output_str
            width_pred= model.width_output_str

        # --------------------------------------------------------------------
        # 3) Call model.compute_loss(...) if you want the final predicted maps
        #    or for synergy with your code. (Not strictly needed for metrics.)
        # --------------------------------------------------------------------
        yc = [ins_mask, qua_mask, cos_mask, sin_mask, wid_mask]
        lossd = model.compute_loss(
            yc=yc,
            mask_pred=mask_pred,
            qua_pred=qua_pred,
            cos_pred=cos_pred,
            sin_pred=sin_pred,
            width_pred=width_pred
        )
        # We won't use 'lossd["loss"]' here for metrics, but you can log it

        # --------------------------------------------------------------------
        # 4) Post-processing: e.g. sigmoid for mask, interpolation if needed
        # --------------------------------------------------------------------
        # Convert to probabilities
        ins_mask_preds = torch.sigmoid(mask_pred)
        qua_preds      = torch.sigmoid(qua_pred)   # if needed
        wid_preds      = torch.sigmoid(width_pred) # if needed

        # If shape mismatch, upsample to image size
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(
                ins_mask_preds, size=image.shape[-2:], mode='bicubic', align_corners=True
            ).squeeze(1)

            qua_preds = F.interpolate(
                qua_preds, size=image.shape[-2:], mode='bicubic', align_corners=True
            ).squeeze(1)
            sin_preds = F.interpolate(
                sin_pred, size=image.shape[-2:], mode='bicubic', align_corners=True
            ).squeeze(1)
            cos_preds = F.interpolate(
                cos_pred, size=image.shape[-2:], mode='bicubic', align_corners=True
            ).squeeze(1)
            wid_preds = F.interpolate(
                wid_preds, size=image.shape[-2:], mode='bicubic', align_corners=True
            ).squeeze(1)
        else:
            sin_preds = sin_pred.squeeze(1)  # or just .squeeze() if shape is [B,1,H,W]
            cos_preds = cos_pred.squeeze(1)

        # --------------------------------------------------------------------
        # 5) For each sample in the batch, inverse warp + compute metrics
        # --------------------------------------------------------------------
        for b_i in range(bs):
            inv_mat = inverse_matrix[b_i]
            (orig_h, orig_w) = ori_sizes[b_i]  # [h, w]

            # CPU numpy arrays
            ins_pred_np = ins_mask_preds[b_i].cpu().numpy()
            qua_pred_np = qua_preds[b_i].cpu().numpy()
            sin_pred_np = sin_preds[b_i].cpu().numpy()
            cos_pred_np = cos_preds[b_i].cpu().numpy()
            wid_pred_np = wid_preds[b_i].cpu().numpy()

            # GT instance mask for IoU
            # If the model's forward doesn't return "target" in the same shape, 
            # we rely on the data batch's ins_mask. But we also might do the same 
            # "inverse warp" for the GT if the training pipeline had it augmented. 
            ins_mask_np = ins_mask[b_i].squeeze().cpu().numpy()

            # Possibly the GT grasp for Jacquard
            grasp_gt = grasp_targets[b_i]

            # Inverse warp predictions + ground truth
            ins_pred_np       = inverse(ins_pred_np, inv_mat, orig_w, orig_h)
            ins_pred_np       = (ins_pred_np > 0.35)
            qua_pred_np       = inverse(qua_pred_np, inv_mat, orig_w, orig_h)
            sin_pred_np       = inverse(sin_pred_np, inv_mat, orig_w, orig_h)
            cos_pred_np       = inverse(cos_pred_np, inv_mat, orig_w, orig_h)
            wid_pred_np       = inverse(wid_pred_np, inv_mat, orig_w, orig_h)
            ins_mask_np       = inverse(ins_mask_np, inv_mat, orig_w, orig_h)

            # IoU
            inter = np.logical_and(ins_pred_np, ins_mask_np)
            union = np.logical_or(ins_pred_np, ins_mask_np)
            iou_val = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou_val)

            # Jacquard
            for n_g, g_count in enumerate(num_grasps):
                grasp_preds, _ = detect_grasps(qua_pred_np, sin_pred_np, cos_pred_np, wid_pred_np, g_count)
                j_val = calculate_jacquard_index(grasp_preds, grasp_gt)
                num_correct_grasps[n_g] += j_val
                num_total_grasps[n_g]   += 1

    # ------------------------------------------------------------------------
    # 6) Gather + compute final metrics
    # ------------------------------------------------------------------------
    iou_array = np.array(iou_list, dtype=np.float32)
    iou_tensor = torch.from_numpy(iou_array).to(device)
    iou_tensor = concat_all_gather(iou_tensor)
    mean_iou = iou_tensor.mean().item()

    # precision thresholds [0.5..0.9]
    prec_vals = []
    for thr in torch.arange(0.5, 1.0, 0.1):
        prec_vals.append((iou_tensor > thr).float().mean().item())

    prec = {}
    for i, th in enumerate(range(5, 10)):
        key = f"Pr@{th*10}"
        prec[key] = prec_vals[i]

    # Jacquard
    j_index = []
    for i_g, g_count in enumerate(num_grasps):
        j_index.append(num_correct_grasps[i_g] / (num_total_grasps[i_g] + 1e-6))

    # Log
    logger.info(
        f"Evaluation: Epoch=[{epoch}/{args.epochs}]  IoU={mean_iou*100:.2f}  "
        f"J_index@1: {j_index[0]*100:.2f}  J_index@5: {j_index[1]*100:.2f}  "
        + "  ".join([f"{k}: {v*100:.2f}" for k,v in prec.items()])
    )

    # Return same format as crog_engine
    return mean_iou, prec, j_index