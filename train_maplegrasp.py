import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial
from collections import OrderedDict

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_API_KEY"] = '99ee90fdefff711f21b8b40a0fac1bdb95da2aa5'


import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.dataset import OCIDVLGDataset,RoboRefGrasp,LiberoRGS,PRISMDataset    
from engine.maplegrasp_engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_maplegrasp
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

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
    torch.multiprocessing.set_start_method('spawn')
    
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    # mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ), join=True)
    
    children = []
    for i in range(args.world_size):
        subproc = mp.Process(target=main_worker, args=(i, args))
        children.append(subproc)
        subproc.start()

    for i in range(args.world_size):
        children[i].join()


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    dist.barrier()

    # build model
    model, param_list = build_crogpp(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info(model)
    logger.info(args)
    
    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()
    
    
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)

    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int(
        (args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    
    if(args.dataset=="OCID-VLG"):

        train_data = OCIDVLGDataset(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='train',
                                version=args.version,)
        val_data = OCIDVLGDataset(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='val',
                                version=args.version,
                                )
        
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
        train_loader = data.DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                    worker_init_fn=init_fn,
                                    sampler=train_sampler,
                                    drop_last=True,
                                    collate_fn=OCIDVLGDataset.collate_fn)
        val_loader = data.DataLoader(val_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    sampler=val_sampler,
                                    drop_last=False,
                                    collate_fn=OCIDVLGDataset.collate_fn)
    
    elif(args.dataset=="RefGraspNet"):

        print("Loading training data")
        train_data = RoboRefGrasp(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='train',
                                version=args.version)
        print("Number of samples in training data: ", len(train_data))
        print("Loading validation data")
        val_data = RoboRefGrasp(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='val',
                                version=args.version)
        print("Number of samples in validation data: ", len(val_data))
        
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)

        train_loader = data.DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                    worker_init_fn=init_fn,
                                    sampler=train_sampler,
                                    drop_last=True,
                                    collate_fn=RoboRefGrasp.collate_fn)
        val_loader = data.DataLoader(val_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    sampler=val_sampler,
                                    drop_last=False,
                                    collate_fn=RoboRefGrasp.collate_fn)
    elif(args.dataset=="LIBERO-SPATIAL"):

        print("Loading training data")
        train_data = LiberoRGS(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='train',
                                )
        print("Number of samples in training data: ", len(train_data))
        print("Loading validation data")
        val_data = LiberoRGS(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split='val',
                                )
        print("Number of samples in validation data: ", len(val_data))
        
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)

        train_loader = data.DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                    worker_init_fn=init_fn,
                                    sampler=train_sampler,
                                    drop_last=True,
                                    collate_fn=LiberoRGS.collate_fn)
        val_loader = data.DataLoader(val_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    sampler=val_sampler,
                                    drop_last=False,
                                    collate_fn=LiberoRGS.collate_fn)
        
    elif(args.dataset=="PRISM"):
        
        print("Loading training data")
        train_data = PRISMDataset(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split=args.train_split,
                                version=args.version)
        print("Number of samples in training data: ", len(train_data))
        print("Loading validation data")
        val_data = PRISMDataset(root_dir=args.root_path,
                                input_size=args.input_size,
                                word_length=args.word_len,
                                split=args.val_split,
                                version=args.version)
        print("Number of samples in validation data: ", len(val_data))
        
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                        shuffle=True)
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)

        train_loader = data.DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                    worker_init_fn=init_fn,
                                    sampler=train_sampler,
                                    drop_last=True,
                                    collate_fn=PRISMDataset.collate_fn)
        val_loader = data.DataLoader(val_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    sampler=val_sampler,
                                    drop_last=False,
                                    collate_fn=PRISMDataset.collate_fn)
        
    best_IoU = 0.0
    best_j_index = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(
                args.resume, map_location=map_location)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            best_j_index = checkpoint["best_j_index"]
            model.load_state_dict(checkpoint['state_dict'],strict=True) # need to add strict=False for stage2
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)
        # evaluation
        if args.stage1 and not args.stage2:
            iou, prec_dict, j_index = validate_without_grasp(val_loader, model, epoch_log, args)
        elif args.stage2 and not args.stage1:
            iou, prec_dict, j_index = validate_with_grasp(val_loader, model, epoch_log, args)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'best_j_index': best_j_index,
                    'prec': prec_dict,
                    'j_index': j_index,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(args.output_dir, "best_iou_model.pth")
                shutil.copyfile(lastname, bestname)
            
            if j_index[0] >= best_j_index:
                best_j_index = j_index[0]
                bestname = os.path.join(args.output_dir, "best_jindex_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)