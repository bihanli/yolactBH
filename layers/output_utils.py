""" Contains functions used to sanitize and prepare the output of Yolact. """

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from data import cfg, mask_type, MEANS, STD, activation_func
from utils.augmentations import Resize
from utils import timer
from .box_utils import crop, sanitize_coordinates

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    dets = det_output[batch_idx]
    net = dets['net']
    dets = dets['detection']

    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing
    
    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4
    #visualize_lincomb=True#........................
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    
    masks   = dets['mask']#19,32
    #for i,j in dets.items():
    #  print(i,j.shape)
    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']#138,138,32
        
        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())
        #print(masks.shape)
        if visualize_lincomb:
            display_lincomb(proto_data, masks)
        #print(proto_data.shape,masks.shape)
        masks = proto_data @ masks.t()#transpose
        masks = cfg.mask_proto_mask_activation(masks)
        #---------------------
        def feature_imshow(inp, title=None):
    
            """Imshow for Tensor."""
    
            #inp = inp.detach().numpy().transpose((1, 2, 0))

            mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
            std = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                  
            inp = std * inp + mean
                  
            inp = np.clip(inp, 0, 1)
            #inp*=255
                  
            plt.pause(0.001)
            return inp
        """
        maskss=masks.permute(2,0,1).cpu()
        #print(maskss.shape)
        masksss=maskss.unsqueeze(1)
        
        ma=[]
        ma.append(masksss[0,:,:,:])
        ma.append(masksss[3,:,:,:])
        ma.append(masksss[4,:,:,:])
        ma.append(masksss[5,:,:,:])
        ma.append(masksss[7,:,:,:])
        ma.append(masksss[11,:,:,:])
        ma.append(masksss[14,:,:,:])
        ma.append(masksss[15,:,:,:])
        masksss=torch.Tensor([item.cpu().detach().numpy() for item in ma]).cuda()
        """
        #thesis,fig8
        """
        masksss = torchvision.utils.make_grid(masksss,nrow=4)
        masksss=masksss.permute(1,2,0).cpu()
        masksss=masksss[:,:,0]
        masksss=masksss.cpu().numpy()
        masksss=Image.fromarray((masksss*255).astype(np.uint16))
        plt.imshow(masksss)
        plt.axis('off')
        plt.savefig("masksss.png")
        #plt.show()
        """
        #-------------------------------
        # Crop masks before upsampling because you know why
        #print(33,masks.shape)
        if crop_masks:
            masks = crop(masks, boxes)
        #print(22,masks.shape)
        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()
        #print(11,masks.shape)
        if cfg.use_maskiou:#T
            with timer.env('maskiou_net'):                
                with torch.no_grad():
                    maskiou_p = net.maskiou_net(masks.unsqueeze(1))
                    
                    #print(maskiou_p.shape,classes)n,13 n
                    maskiou_p = torch.gather(maskiou_p, dim=1, index=classes.unsqueeze(1)).squeeze(1)
                    #print(maskiou_p,scores.shape)n,n
                    if cfg.rescore_mask:#T
                        if cfg.rescore_bbox:#F
                            scores = scores * maskiou_p
                        else:
                            scores = [scores, scores * maskiou_p]
        
        #print(masks.shape)
        # Scale masks up to the full image
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)
        #thesis,fig9a
        """
        #print(masks.shape)
        masks_s=masks.unsqueeze(1)
        #print(masks_s.shape)
        masks_s=torchvision.utils.make_grid(masks_s,nrow=4,padding=1)
        #print(masks_s.shape)
        masks_s=masks_s.cpu().numpy()
        masks_s=masks_s.transpose((1,2,0))
        masks_s=masks_s[:,:,0]
        #print(masks_s.shape)
        masks_s=Image.fromarray((masks_s*255).astype(np.uint16))
        plt.imshow(masks_s)
        plt.axis('off')
        plt.savefig("masks_s.png")
        #plt.show()
        """
        #thesis,fig9b
        """
        mm=masks.cpu().numpy()
        mmm=mm.sum(axis=0)
        #print(mmm.shape)
        mmm=Image.fromarray((mmm*255).astype(np.uint16))
        plt.imshow(mmm)
        plt.axis('off')
        plt.savefig("mmm.png")
        #plt.show()
        """
        # Binarize the masks
        masks.gt_(0.5)

    
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.long()

    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    return classes, scores, boxes, masks


    


def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To BRG

    if cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    elif cfg.backbone.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)
        
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)
    #print(img_numpy.shape)
    return cv2.resize(img_numpy, (w,h))


def display_lincomb(proto_data, masks):
    out_masks = torch.matmul(proto_data, masks.t())
    # out_masks = cfg.mask_proto_mask_activation(out_masks)

    for kdx in range(16):
        jdx = kdx 
        
        
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        #x=np.arange(-0.5,32.5,1)
        #y=[0]*33
        plt.bar(list(range(idx.shape[0])), coeffs[idx],1,edgecolor='k')
        #plt.plot(x,y,'k',linewidth=0.5)
        plt.axis([-0.5,31.5,-1,1])
        plt.savefig("bar"+str(jdx)+".png")
        #plt.show()
        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4,8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
        arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])
        test = torch.sum(proto_data, -1).cpu().numpy()

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = running_total
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    running_total_nonlin = (1/(1+np.exp(-running_total_nonlin)))

                arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (running_total_nonlin > 0.5).astype(np.float)

        plt.imshow(arr_img)
        plt.axis('off')
        plt.savefig("arr_img"+str(jdx)+".png")
        #plt.show()
        plt.imshow(arr_run)
        plt.axis('off')
        plt.savefig("arr_run"+str(jdx)+".png")
        #plt.show()
        plt.imshow(test)
        plt.axis('off')
        plt.savefig("test"+str(jdx)+".png")
        #plt.show()
        plt.imshow(out_masks[:, :, jdx].cpu().numpy())
        plt.axis('off')
        plt.savefig("out_masks"+str(jdx)+".png")
        #plt.show()
