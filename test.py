import argparse
import os
import warnings
import torch.nn as nn
import cv2
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data as data
import torch.utils.data
from loguru import logger
from model.clipseg import CLIPSEG
from model.lgrconvnet import LGRCONVNET
import utils.config as config
from engine.crog_engine import inference_with_grasp,validate_with_grasp
from model import build_crog,build_maplegrasp
from utils.dataset import OCIDVLGDataset,RoboRefGrasp,LiberoRGS
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    print("Visualize: ", args.visualize)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build model
    model,_ = build_crogpp(args)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[0],find_unused_parameters=True)

    
    logger.info(model)
    
    save_path = os.path.join("./results", args.output_dir.replace("exp/",""))
    os.makedirs(save_path, exist_ok=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(
                args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            #best_IoU = checkpoint.get('best_IoU', 0)
            #best_j_index = checkpoint.get('best_j_index', 0)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # build dataset & dataloader - OCIDVLG
    
    test_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='test',
                            version=args.version,
                            exp_name=args.exp_name)
    test_sampler = data.distributed.DistributedSampler(test_data, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              collate_fn=OCIDVLGDataset.collate_fn)

    
    #build dataset & dataloader - RoboRefGrasp
    # test_data = RoboRefGrasp(root_dir='/hdddata/vineet/RefGraspNet/',
    #                     split=args.test_split,
    #                     input_size=args.input_size,
    #                     word_length=args.word_len,
    #                     with_segm_mask=True,
    #                     with_grasp_masks=True,
    #                     version=args.version,
    #                     exp_name=args.exp_name)

    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=16,
    #                                           shuffle=False,
    #                                           num_workers=1,
    #                                           pin_memory=True,
    #                                           collate_fn=RoboRefGrasp.collate_fn)
    
    # test_data = LiberoRGS(root_dir=args.root_path,
    #                     split=args.test_split,
    #                     input_size=args.input_size,
    #                     word_length=args.word_len,
    #                     with_segm_mask=True,
    #                     with_grasp_masks=True,
    #                     exp_name=args.exp_name)

    # test_loader = torch.utils.data.DataLoader(test_data,
    #                                           batch_size=16,
    #                                           shuffle=False,
    #                                           num_workers=1,
    #                                           pin_memory=True,
    #                                           collate_fn=LiberoRGS.collate_fn)

    # inference
    #inference_with_grasp(test_loader, model, args)
    validate_with_grasp(test_loader, model, args.start_epoch,args)
    #validate_without_grasp(test_loader, model, args.start_epoch,args)


if __name__ == '__main__':
    main()
