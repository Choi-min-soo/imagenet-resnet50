import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import os
import wandb
import math

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(warmup_epochs, 1))
        progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                p.sam_e_w = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(p.sam_e_w)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self): raise NotImplementedError
    def zero_grad(self): self.base_optimizer.zero_grad()
    def _grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        norms = [p.grad.norm(p=2).to(device) for group in self.param_groups for p in group['params'] if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2)

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.decay).add_(v.data, alpha=1 - self.decay)

    def apply_shadow(self):
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema, strict=False)

    def restore(self):
        self.model.load_state_dict(self.backup, strict=False)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(images, labels, alpha=1.0):
    lam = np.clip(np.random.beta(alpha, alpha), 1e-4, 1. - 1e-4)
    rand_index = torch.randperm(images.size(0)).to(images.device)
    target_a, target_b = labels, labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    area = (bbx2 - bbx1) * (bby2 - bby1)
    if area > 0:
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1. - area / (images.size(-1) * images.size(-2))
    else:
        lam = 1.0
    return images, target_a, target_b, lam

def mixup(images, labels, alpha=1.0):
    lam = np.clip(np.random.beta(alpha, alpha), 1e-4, 1. - 1e-4)
    index = torch.randperm(images.size(0)).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam

def train(cfg, model, train_loader, val_loader, device, rank=0):
    is_master = (rank == 0)
    if is_master:
        wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["name"], config=cfg)

    tr_cfg, opt = cfg["training"], cfg["optimizer"]
    criterion = LabelSmoothingLoss(0.1)
    scaler = GradScaler() if tr_cfg.get("use_amp", False) else None
    base_optimizer = SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=opt["lr"], momentum=opt["momentum"], weight_decay=opt["weight_decay"]) if tr_cfg.get("use_sam", False) else SGD(model.parameters(), lr=opt["lr"], momentum=opt["momentum"], weight_decay=opt["weight_decay"])

    base_opt = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
    scheduler = get_warmup_cosine_scheduler(
        base_opt,
        warmup_epochs=tr_cfg.get("warmup_epochs", 0),
        max_epochs=tr_cfg["max_epochs"]
    )
    ema = EMA(model)
    best_acc = 0.0

    for epoch in range(tr_cfg["max_epochs"]):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}]") if is_master else train_loader

        for i, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            p = np.random.rand()
            cutmix_prob, mixup_prob = tr_cfg.get("cutmix_prob", 0.0), tr_cfg.get("mixup_prob", 0.0)

            if p < cutmix_prob:
                images, ta, tb, lam = cutmix(images, labels)
            elif p < cutmix_prob + mixup_prob:
                images, ta, tb, lam = mixup(images, labels)
            else:
                ta, tb, lam = labels, labels, 1.0

            def run_model():
                outputs = model(images)
                return lam * criterion(outputs, ta) + (1 - lam) * criterion(outputs, tb), outputs

            if scaler:
                with autocast():
                    loss, outputs = run_model()
                scaler.scale(loss).backward()
            else:
                loss, outputs = run_model()
                loss.backward()

            if tr_cfg.get("use_sam", False):
                optimizer.first_step(zero_grad=True)
                if scaler:
                    with autocast():
                        loss2, _ = run_model()
                    scaler.scale(loss2).backward()
                    scaler.step(optimizer.base_optimizer)
                    scaler.update()
                else:
                    loss2, _ = run_model()
                    loss2.backward()
                    optimizer.base_optimizer.step()
                optimizer.second_step(zero_grad=True)
            else:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            ema.update()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if is_master and (i + 1) % tr_cfg["print_interval"] == 0:
                loop.set_postfix(loss=loss.item(), acc=correct / total)

        scheduler.step()
        if is_master:
            wandb.log({"train/loss": total_loss / total, "train/acc": correct / total, "epoch": epoch})

        ema.apply_shadow()
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast() if scaler else torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        ema.restore()
        val_acc = val_correct / val_total
        if is_master:
            wandb.log({"val/loss": val_loss / val_total, "val/acc": val_acc, "epoch": epoch})
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_name = cfg["wandb"]["name"] + ".pt"
                ckpt_path = os.path.join(cfg["training"]["save_dir"], ckpt_name)
                torch.save(model.state_dict(), ckpt_path)
                print(f"💾 Saved best model at {ckpt_path} (acc={best_acc:.4f})")