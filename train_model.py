import os
import os.path as osp
import argparse
import numpy as np
import logging

# os.environ['CUDA_VISIBLE_DEVICES']='2'
os.environ["CUDA_VISIBLE_DEVICES"]  = '2'
import torch
import torch.distributed as dist

import mmcv
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import DistEvalHook as BasicDistEvalHook
from mmcv.runner import EpochBasedRunner, OptimizerHook, DistSamplerSeedHook
from mmcv.runner import build_optimizer, set_random_seed, get_dist_info, init_dist
from mmcv.utils import get_logger

from datasets.builder import build_dataset, build_dataloader
from model.builder import build_model,build_recognizer

from utils.misc import cache_checkpoint
from mmcv.runner import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train")

    # parser.add_argument('--config', default='/home/shr/code/gcn_vivit/.vscode/train/config.py')
    parser.add_argument('--config', default='/home/shy/code_hik/video_model/config/ske_gcn/config_aagcn_vivit.py')

    parser.add_argument('--launcher', default='pytorch', help='job launcher')

    parser.add_argument('--non-dist', action='store_true', default='True', help='whether to use distributed skeleton extraction')

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()

    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(args.local_rank)

    return args


def init_random_seed(seed=None, device='cuda'):
    if seed is not None:
        return seed
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed
    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)
    return random_num.item()


def get_root_logger(log_file=None, log_level=logging.INFO):
    return get_logger(__name__.split('.')[0], log_file, log_level)


class DistEvalHook(BasicDistEvalHook):
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP@', 'Recall@'
    ]
    less_keys = ['loss']

    def __init__(self, *args, save_best='auto', **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)



def train_model(model, dataset, cfg, meta=None):

    logger = get_root_logger(log_level=cfg.log_level)

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    
    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]
    # data_loaders = []
    # train_dataloader = build_dataloader(dataset, **dataloader_setting)

    find_unused_parameters = cfg.get('find_unused_parameters', False)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters
    )

    optimizer = build_optimizer(model, cfg.optimizer)

    Runner = EpochBasedRunner
    runner = Runner(
        model=model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta
    )

    if 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    
    runner.register_training_hooks(
        lr_config=cfg.lr_config, 
        optimizer_config=optimizer_config, 
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config,
        momentum_config=cfg.get('momentum_config', None)
    )
    runner.register_hook(DistSamplerSeedHook())

    eval_cfg = cfg.get('evaluation', {})
    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                                **cfg.data.get('val_dataloader', {}))
    val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
    eval_hook = DistEvalHook(val_dataloader, **eval_cfg)
    runner.register_hook(eval_hook)

    data_loaders.append(val_dataloader)
    # data_loaders.append(train_dataloader)
    

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    # os.environ["CUDA_VISIBLE_DEVICES"]  = '1'
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    datasets = [build_dataset(cfg.data.train)]
    # cfg.workflow = cfg.get('workflow', [('train', 1)])
    cfg.workflow = cfg.get('workflow', [('train', 1),('val', 1)])

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')
    os.environ["WORLD_SIZE"]="1"
    os.environ["MASTER_ADDR"]="localhost"
    os.environ['MASTER_PORT'] = '22077'
    init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

    seed = init_random_seed()
    set_random_seed(seed)
    cfg.seed = seed
    meta = dict()
    meta['seed'] = seed
    meta['config_name'] = os.path.basename(args.config)

    # pretrained_path = '/home/shy/code_hik/video_model/.vscode/resources/pretrained_model.pth'
    # model = build_model(cfg.model)
    # model.load_state_dict(torch.load(pretrained_path))

    config = mmcv.Config.fromfile(args.config)
    model = build_recognizer(config.model)
    # model = build_model(config.model)
    # checkpoint_path = '/home/shihr/code/video_model/.vscode/resources/pretrained_model.pth'
    # checkpoint = cache_checkpoint(checkpoint_path)
    # load_checkpoint(model, checkpoint, map_location='cpu')

    # freeze
    if cfg.mode == 'finetune':
        for param in model.backbone.parameters():
            param.requires_grad = False

    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:23422', world_size=1, rank=0)
    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    dist.barrier()
    train_model(model=model, dataset=datasets, cfg=cfg, meta=meta)
    dist.barrier()

if __name__ == '__main__':
    main()