
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

from backbone import build_backbone
from dataloader import get_loader
from wrapper import train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

def cleanup_ddp():
    dist.destroy_process_group()

def main(cfg, local_rank):
    print(f"🔍 RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, args.local_rank={local_rank}")
    sys.stdout.flush()

    set_seed(cfg.get("seed", 42))
    print(f"[{local_rank}] 🛠️ Setting up DDP...")
    sys.stdout.flush()

    setup_ddp(local_rank)
    print(f"[{local_rank}] ✅ DDP setup done")
    sys.stdout.flush()

    print(f"[{local_rank}] 🧠 Building model...")
    sys.stdout.flush()

    device = torch.device("cuda", local_rank)
    model = build_backbone(cfg).to(device)
    print(f"[{local_rank}] ✅ model.to(device) done")
    sys.stdout.flush()

    print(f"[{local_rank}] ⏳ wrapping DDP now")
    sys.stdout.flush()

    model = DDP(model, device_ids=[local_rank])
    print(f"[{local_rank}] ✅ Model DDP wrapped")
    sys.stdout.flush()

    if cfg["training"].get("resume_from"):
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        model.module.load_state_dict(torch.load(cfg["training"]["resume_from"], map_location=map_location))
        if local_rank == 0:
            print(f"✅ Loaded weights from {cfg['training']['resume_from']}")

    print(f"[{local_rank}] 📦 Loading data...")
    train_loader = get_loader(cfg, mode="train", ddp=True, local_rank=local_rank)
    val_loader = get_loader(cfg, mode="val", ddp=True, local_rank=local_rank)
    print(f"[{local_rank}] ✅ Data loaded")

    print(f"[{local_rank}] 🚀 Entering training loop")
    train(cfg, model, train_loader, val_loader, device, rank=local_rank)
    cleanup_ddp()

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    local_rank = int(os.environ["LOCAL_RANK"])  # ✅ 핵심 수정
    main(cfg, local_rank)
