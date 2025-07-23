import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import get_dataset

def get_loader(cfg, mode="train", ddp=False, local_rank=0):
    dataset = get_dataset(cfg, mode)
    ds_cfg = cfg["dataset"]
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
        shuffle = False
    else:
        sampler = None
        shuffle = (mode == "train")

    return DataLoader(
        dataset,
        batch_size=ds_cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=ds_cfg.get("pin_memory", True),
        persistent_workers=True
    )