import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import imageio
import os
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
import cv2
import errno

def save_results(save_dir, epoch, step, unnorm_videos, outputs, p_x, bool_masked_pos):
    # Number of patches
    t, h, w = 8, 14, 14

    # Tubelet size
    p0, p1, p2 = 2, 16, 16

    # unnorm_videos.shape = B, C, T, H, W
    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=p0, p1=p1, p2=p2)

    # outputs.shape = B, C, T, H, W
    outputs = rearrange(outputs, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=t, h=h, w=w, p0=p0, p1=p1, p2=p2)
    outputs = rearrange(outputs, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=p0, p1=p1, p2=p1)
    outputs = outputs*(videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()+ 1e-6) + videos_squeeze.mean(dim=-2, keepdim=True)
    outputs = rearrange(outputs, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)',t=8, h=14, w=14, p0=2, p1=p1, p2=p2)
    outputs = (outputs-torch.min(outputs, dim=0, keepdim=True).values)/(torch.max(outputs, dim=0, keepdim=True).values-torch.min(outputs, dim=0, keepdim=True).values)


    # p.shape = B t h w
    p_x = rearrange(p_x, 'b (t h w) -> b t h w', t=t, h=h, w=w)
    p_x = repeat(p_x, 'b t h w -> b t (h p1) (w p2)', p1=p1, p2=p2)

    # mask.shape = B
    mask = rearrange(bool_masked_pos, 'b (t h w) -> b t h w', t=t, h=h, w=w)
    mask = repeat(mask, 'b t h w -> b t (h p1) (w p2)', p1=p1, p2=p2)

    gif_folder = os.path.join(save_dir, 'output_gifs' ,str(epoch))
    img_folder = os.path.join(save_dir, 'output_imgs' ,str(epoch))
    try:
        os.makedirs(gif_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    try:
        os.makedirs(img_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    for b in range(5):
        combined_gif = []
        gif_path = os.path.join(gif_folder, str(b)+'.gif')

        p_all = torch.squeeze(p_x[b,:,:,:]).detach().cpu().numpy()
        for idx_t in range(p_x.shape[1]):
            #video
            v = (255.*torch.squeeze(unnorm_videos[b,:,2*idx_t,:,:])).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            
            #pred
            pred = (255.*torch.squeeze(outputs[b,:,2*idx_t,:,:])).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            
            #multinomial
            p = (255.*(p_all[idx_t]-np.amin(p_all))/(np.amax(p_all)-np.amin(p_all)+1e-9)).astype(np.uint8)
            p = cv2.applyColorMap(p, cv2.COLORMAP_JET)
            p  = p[:, :, ::-1]

            #mask
            m = (255*torch.squeeze(mask[b,idx_t,:,:])).detach().cpu().numpy().astype(np.uint8)
            m = np.expand_dims(m, 2)
            m = np.repeat(m, 3, axis=-1)

            #error
            E = (torch.squeeze(unnorm_videos[b,:,2*idx_t,:,:])-torch.squeeze(outputs[b,:,2*idx_t,:,:]))**2
            E = torch.mean(E, dim=0, keepdim=False).detach().cpu().numpy()
            E = (255.*(E-np.amin(E))/(np.amax(E)-np.amin(E)+1e-4)).astype(np.uint8)
            e = cv2.applyColorMap(E, cv2.COLORMAP_JET)
            e  = e[:, :, ::-1]

            #combined
            combined = np.concatenate(( np.clip(v,0,255), 
                                        np.clip(pred,0,255), 
                                        np.clip(e,0,255), 
                                        np.clip(p,0,255), 
                                        np.clip(255-m,0,255)
                                        ), axis=1)
            combined_gif.append(combined)

            # Save individial imgs
            p  = p[:, :, ::-1]
            e  = e[:, :, ::-1]
            cv2.imwrite(os.path.join(img_folder, str(b)+'_gt_video_f'+str(idx_t)+'.png'), v)
            cv2.imwrite(os.path.join(img_folder, str(b)+'_pre_video_f'+str(idx_t)+'.png'), pred)
            cv2.imwrite(os.path.join(img_folder, str(b)+'_error_f'+str(idx_t)+'.png'), e)
            cv2.imwrite(os.path.join(img_folder, str(b)+'_p_f'+str(idx_t)+'.png'), p)
            cv2.imwrite(os.path.join(img_folder, str(b)+'_mask_f'+str(idx_t)+'.png'), 255-m)

        #Save combined gif file
        imageio.mimsave(gif_path, combined_gif)


