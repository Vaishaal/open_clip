import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from .distributed import is_master
from .zero_shot import zero_shot_eval

from contextlib import suppress
import wandb
import logging


def gather_features(
        features0_features, features1_features,
        local_loss=False, gather_with_grad=False, rank=0, world_size=1, horovod=False):
    if horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_features0_features = hvd.allgather(features0_features)
            all_features1_features = hvd.allgather(features1_features)
        else:
            with torch.no_grad():
                all_features0_features = hvd.allgather(features0_features)
                all_features1_features = hvd.allgather(features1_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features0_features = list(all_features0_features.chunk(world_size, dim=0))
                gathered_features1_features = list(all_features1_features.chunk(world_size, dim=0))
                gathered_features0_features[rank] = features0_features
                gathered_features1_features[rank] = features1_features
                all_features0_features = torch.cat(gathered_features0_features, dim=0)
                all_features1_features = torch.cat(gathered_features1_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features0_features = torch.cat(torch.distributed.nn.all_gather(features0_features), dim=0)
            all_features1_features = torch.cat(torch.distributed.nn.all_gather(features1_features), dim=0)
        else:
            gathered_features0_features = [torch.zeros_like(features0_features) for _ in range(world_size)]
            gathered_features1_features = [torch.zeros_like(features1_features) for _ in range(world_size)]
            dist.all_gather(gathered_features0_features, features0_features)
            dist.all_gather(gathered_features1_features, features1_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features0_features[rank] = features0_features
                gathered_features1_features[rank] = features1_features
            all_features0_features = torch.cat(gathered_features0_features, dim=0)
            all_features1_features = torch.cat(gathered_features1_features, dim=0)
    return all_features0_features, all_features1_features



class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.horovod = horovod
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features, 
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            self.labels[device] = labels
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            rank=0,
            world_size=1,
            horovod=False,
            temperature=0.1
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.horovod = horovod
        self.prev_num_logits = 0
        self.labels = {}
        self.tau = temperature

    def forward(self, aug1_embed, aug2_embed):
        device = aug1_embed.device
        q_a = aug1_embed
        q_b = aug2_embed
        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)
        local_batch_size = q_a.size(0)
        k_a, k_b = gather_features(
            aug1_embed, aug2_embed, 
            self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.horovod)

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau
        num_logits = logits_aa.shape[0]

        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            self.labels[device] = labels
            total_batch_size = local_batch_size * self.world_size
            all_labels = torch.cat([self.labels[x] for x in self.labels.keys()], dim=0)
            self.masks = F.one_hot(all_labels, total_batch_size) * 1e9
        else:
            labels = self.labels[device]

        logits_aa = logits_aa - self.masks
        logits_bb = logits_bb - self.masks
        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples
        return loss

class SLIPLoss(nn.Module):
    def __init__(self, ssl_loss, clip_loss, ssl_scale):
        super().__init__()
        self.clip_loss = clip_loss
        self.ssl_loss = ssl_loss
        self.ssl_scale = ssl_scale

    def forward(self, image_features, text_features, logit_scale, aug1_embed, aug2_embed):
        clip_loss = self.clip_loss(image_features, text_features, logit_scale)
        ssl_loss = self.ssl_loss(aug1_embed, aug2_embed)
        return clip_loss + self.ssl_scale * ssl_loss

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    os.environ["WDS_EPOCH"] = str(epoch)

    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    model.train()
    clip_loss = ClipLoss(args.local_loss, args.gather_with_grad, args.rank, args.world_size, args.horovod)
    if args.SLIP:
        simclr_loss = SIMCLRLoss(args.local_loss, args.gather_with_grad, args.rank, args.world_size, args.horovod)
        loss = SLIPLoss(simclr_loss, clip_loss, args.ssl_scale)
    else:
        loss = clip_loss

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch = dataloader.num_batches
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts, aug1, aug2 = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        aug1 = aug1.to(device=device, non_blocking=True)
        aug2 = aug2.to(device=device, non_blocking=True)

        data_time = time.time() - end
        optimizer.zero_grad()

        with autocast():
            if args.SLIP:
                image_features, text_features, logit_scale, aug1_embed, aug2_embed = model(images, texts, aug1, aug2)
                total_loss = loss(image_features, text_features, logit_scale, aug1_embed, aug2_embed)
            else:
                image_features, text_features, logit_scale = model(images, texts)
                total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        logit_scale = unwrap_model(model).logit_scale
        with torch.no_grad():
            logit_scale.clamp_(0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) and (i % 100) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            loss_scalar = total_loss.item()
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss_scalar:.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {logit_scale_scalar:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": loss_scalar,
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None):
    if not is_master(args):
        return
    device = torch.device(args.device)
    metrics = {}
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    if 'val' in data:
        dataloader = data['val'].dataloader

        cumulative_loss = 0.0
        num_elements = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for batch in dataloader:
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                image_features, text_features, logit_scale = model(images, texts)
                all_image_features.append(image_features)
                all_text_features.append(text_features)
                logit_scale = logit_scale.mean()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                total_loss = (
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                ) / 2
                cumulative_loss += total_loss * batch_size
                num_elements += batch_size

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale
            )
            loss = cumulative_loss / num_elements
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
            )

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
