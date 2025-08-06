import argparse
import os
import random
import sys
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.getcwd()))

from backbone import get_model, get_output_dim
from data.loaders import DataLoaderX
from data.transform import transform_image
from training import get_trainer
from utils.logging import TrainingLogger

from torch.utils.data import DataLoader

from data import get_dataset

torch.backends.cudnn.benchmark = True

os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

def main(cfg):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) # args.local_rank
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    # Logging
    TrainingLogger(local_rank, cfg.output_path)

    # testset a list of datasets, create a sampler for each, and a dataloader for each
    testset = get_dataset(local_rank, **cfg)
    test_sampler = []
    test_dataloader = []
    for test in testset:
        test_dataloader.append(DataLoader(dataset=test, batch_size=cfg.batch_size,
            pin_memory=True, drop_last=False, num_workers=0
        ))

    # Model
    model, preprocess = get_model(local_rank, **cfg)

    if cfg.model_name == 'siglip' or cfg.model_name == 'blip':
         model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank], find_unused_parameters=False)
    if cfg.use_lora or cfg.train_scratch:
        model.train()

    # Training
    model_trainer = get_trainer(
        rank=local_rank,
        world_size=world_size,
        model_name=cfg.model_name,
        model=model,
        preprocess=preprocess,
        config=cfg,
        test_dataloader=test_dataloader,
        test_sampler=test_sampler
    )
    model_trainer.start_training()

    if local_rank == 0:
        destroy_process_group()