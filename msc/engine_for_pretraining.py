import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import msc.utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import scipy.io
import os
from msc.save_to_gifs import save_results

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, save_dir=None):
    model.train()
    metric_logger = msc.utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', msc.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', msc.utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction="none")

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, _ = batch
        videos = videos.to(device, non_blocking=True)
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            # labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        with torch.cuda.amp.autocast():
            outputs, p_x, vis_idx, bool_masked_pos = model(videos)

            # labels (lbls in the original order)
            mask_labels = videos_patch[bool_masked_pos].reshape(B, -1, C) # vis_labels = videos_patch[~bool_masked_pos].reshape(B, -1, C)

            # outputs ([vis, mask] order)
            mask_outputs = outputs[:, -mask_labels.shape[1]:] # vis_outputs = outputs[:, :vis_labels.shape[1]]
            
            # losses
            # l_r -> B, N_m (for all tokens)
            mask_l_r = torch.mean(loss_func(input=mask_outputs, target=mask_labels), dim=-1) #B, N_m # vis_l_r = torch.mean(loss_func(input=vis_outputs, target=vis_labels), dim=-1) #B, N_m

            # l_s -> B, N_m
            l_s =torch.zeros(videos.shape[0], ).to(mask_l_r.device)
            for i in range(p_x.shape[0]):
                # distribution
                m = torch.distributions.categorical.Categorical(probs=p_x[i])
                
                # log-probabilities
                log_probs = m.log_prob(torch.arange(0, p_x.shape[1], 1).to(p_x.device)) # 1, N_m
                
                # visible log-probs
                # vis_log_probs = log_probs[~bool_masked_pos[i]]
                mask_log_probs = log_probs[bool_masked_pos[i]]

                # expected log reconstruction loss
                # we need to select tokens that maximize the reconstruction error, so (-) sign
                # l_s[i] = -torch.mean(vis_log_probs)*torch.mean(mask_l_r[i].detach())
                l_s[i] = -torch.mean(mask_log_probs*mask_l_r[i].detach())
                
            # total loss
            m_l_r = torch.mean(mask_l_r)
            m_l_s = 1e-4*torch.mean(l_s)
            loss = m_l_r + m_l_s

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_reconstruction=m_l_r.item(), head="loss")
            log_writer.update(loss_sampling=m_l_s.item(), head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()
        
        # Save results
        if (epoch % 1 ==0) and (step == 0):
            if str(unnorm_videos.device) == 'cuda:0':
                print('saving results ...')
                save_results(save_dir, epoch, step, unnorm_videos, outputs, p_x, bool_masked_pos)
            
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
