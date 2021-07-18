import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage, make_net

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn



class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

prior_cache = defaultdict(lambda: None)

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.num_classes = cfg.num_classes#class
        self.mask_dim    = cfg.mask_dim # Defined by Yolact k coefficient
        self.num_priors  = sum(len(x)*len(scales) for x in aspect_ratios)#9=3*3 24,30,38,1,0.5,2
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index#0~4
        self.num_heads   = cfg.num_heads # Defined by Yolact p3~p7(5)
        
        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:#F,1=1
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:#F
            in_channels += self.mask_dim
        
        if parent is None:#none,predictionmodule
            if cfg.extra_head_net is None:#[(256, 3, {'padding': 1})]
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)
                #Sequential((0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  (1): ReLU(inplace=True)), 256
                
            if cfg.use_prediction_module:#F
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)
            
            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)#256,9,{'kernel_size': 3, 'padding': 1}
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)#256,9*classes
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)#256,9*32
            
            if cfg.use_mask_scoring:#F
                self.score_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff:#F
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))
            
            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]#000
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate: #1=1&F
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)
        
        self.aspect_ratios = aspect_ratios #[[1, 0.5, 2]]
        self.scales = scales #[24.0, 30.238105197476955, 38.097625247236785]*2.*4.*8.*16

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]#TFFFF
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if cfg.extra_head_net is not None:#T
            x = src.upfeature(x)#self,parent[0]
            
        if cfg.use_prediction_module:#F
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)
        
        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        
        if cfg.eval_mask_branch:#T
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:#F
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_mask_scoring:#F
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        if cfg.use_instance_coeff:#F
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)    

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:#F
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:#T
            if cfg.mask_type == mask_type.direct:#F1:0
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:#T1:1
                mask = cfg.mask_proto_coeff_activation(mask)#tanh
                
                if cfg.mask_proto_coeff_gate:#F
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:#F&1=1
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode='constant', value=0)
        
        priors = self.make_priors(conv_h, conv_w, x.device)#69,35,18,9,5
        
        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
        #print(preds['loc'].shape,preds['conf'].shape,preds['mask'].shape,preds['priors'].shape)
        if cfg.use_mask_scoring:#F
            preds['score'] = score

        if cfg.use_instance_coeff:#F
            preds['inst'] = inst
        
        return preds

    def make_priors(self, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = (conv_h, conv_w)

        with timer.env('makepriors'):
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):#69,35,18,9,5
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for ars in self.aspect_ratios:#1,0.5,2
                        for scale in self.scales:#[24,30,38]*1.*2.*4.*8.*16
                            for ar in ars:#1,0.5,2
                                if not cfg.backbone.preapply_sqrt:#FF=T
                                    ar = sqrt(ar)

                                if cfg.backbone.use_pixel_scales:#T
                                    w = scale * ar / cfg.max_size#550
                                    h = scale / ar / cfg.max_size
                                else:#F
                                    w = scale * ar / conv_w
                                    h = scale / ar / conv_h
                                
                                # This is for backward compatability with a bug where I made everything square by accident
                                if cfg.backbone.use_square_anchors:#F
                                    h = w

                                prior_data += [x, y, w, h]
                
                self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach()#headp3p4p5p6p7[42849, 4][11025, 4][2916, 4][729, 4][225, 4]
                self.priors.requires_grad = False
                self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)#550
                self.last_conv_size = (conv_w, conv_h)#69,35,18,9,5
                prior_cache[size] = None
            elif self.priors.device != device:#F
                # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
                if prior_cache[size] is None:
                    prior_cache[size] = {}
                
                if device not in prior_cache[size]:
                    prior_cache[size][device] = self.priors.to(device)

                self.priors = prior_cache[size][device]
        
        return self.priors

class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()
        #[512,1024,2048]
        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])
        
        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0  #T
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample: #T
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample) #2
            ])
        
        self.interpolation_mode     = cfg.fpn.interpolation_mode #bilinear
        self.num_downsample         = cfg.fpn.num_downsample #2
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample #T
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers #F
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers #T
        
    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """
        
        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)
        
        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)#3
        for lat_layer in self.lat_layers:
            j -= 1
            
            if j < len(convouts) - 1:
                
                _, _, h, w = convouts[j].size()#35,69
                
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)#插植18,35,69
            
            x = x + lat_layer(convouts[j])#上採樣
            out[j] = x
            
        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])
            
            if self.relu_pred_layers:#T
                F.relu(out[j], inplace=True)
            
        cur_idx = len(out)#3
        
        # In the original paper, this takes care of P6
        if self.use_conv_downsample:#T
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
                
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:#F
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)
        
        return out

class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p



class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self, only_last_layer=False):
        super().__init__()
        
        self.only_last_layer = only_last_layer
        self.backbone = construct_backbone(cfg.backbone)  #cfg=yolact_base_config backbone=resnet101_backbone

        if cfg.freeze_bn:
            self.freeze_bn()
        
        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:  #1 0
            cfg.mask_dim = cfg.mask_size**2
            
        elif cfg.mask_type == mask_type.lincomb: #1=1
            if cfg.mask_proto_use_grid: #False
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src #0
            
            if self.proto_src is None: in_channels = 3 #F
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features #256
            else: in_channels = self.backbone.channels[self.proto_src]  #F
            in_channels += self.num_grids #256
            
            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False) #,32=256,c*3+interpolate+c*2,32[(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
            
            if cfg.mask_proto_bias:#F
                cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers #[1,2,3]
        src_channels = self.backbone.channels #[256, 512, 1024, 2048]
        
        if cfg.use_maskiou:#T
            self.maskiou_net = FastMaskIoUNet()

        if cfg.fpn is not None:#T
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))#3+2=[0,1,2,3,4]
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)#256*5
            


        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)#5
        
        for idx, layer_idx in enumerate(self.selected_layers):#[0,1,2,3,4]
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            
            if cfg.share_prediction_module and idx > 0: #T&0~4
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],#1,0.5,2
                                    scales        = cfg.backbone.pred_scales[idx],#24,48,96,192,384*2^(0,1/3,2/3)
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)
        
        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:#F
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg.use_semantic_segmentation_loss:#T
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)#256
        
        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, #class,0,200
            conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh) #0.05,0.5
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        # Uncomment this in normal conditions
        # self.load_state_dict(state_dict)
        # Added this for fine-tuning. Comment this in normal conditions.
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()
    
    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
    
    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        
        with timer.env('backbone'):
            outs = self.backbone(x)
            
        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers]#123
                outs = self.fpn(outs)#p3~p7
        
        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:#11T
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]#F0p3
                
                if self.num_grids > 0:#0F
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)
                proto_out = self.proto_net(proto_x)#c*3插cc
                proto_out = cfg.mask_proto_prototype_activation(proto_out)#relu
                #----------------------------------------------
                
                def feature_imshow(inp, title=None):
    
                  """Imshow for Tensor."""
    
                  inp = inp.cpu().detach().numpy().transpose((1, 2, 0))

                  mean = np.array([0.5, 0.5, 0.5])
    
                  std = np.array([0.5, 0.5, 0.5])
                  
                  inp = std * inp + mean
                  
                  inp = np.clip(inp, 0, 1)
                  #inp*=255
                  
                  plt.pause(0.001)
                  return inp  # pause a bit so that plots are updated
                #print(proto_out.shape)
                feature_ouput1=proto_out.transpose(1,0).cpu()
                #thesis,fig1
                """
                feature_ouput=[]
                print(feature_ouput1.shape)
                feature_ouput.append(feature_ouput1[3,:,:,:])
                feature_ouput.append(feature_ouput1[4,:,:,:])
                feature_ouput.append(feature_ouput1[7,:,:,:])
                feature_ouput.append(feature_ouput1[8,:,:,:])
                feature_ouput.append(feature_ouput1[22,:,:,:])
                feature_ouput.append(feature_ouput1[24,:,:,:])
                feature_ouput.append(feature_ouput1[29,:,:,:])
                feature_ouput.append(feature_ouput1[31,:,:,:])
                feature_ouput1=torch.Tensor([item.cpu().detach().numpy() for item in feature_ouput]).cuda()
                #print(fea.shape)
                
                out = torchvision.utils.make_grid(feature_ouput1,nrow=4)
                """
                #thesis,fig7
                """
                out = torchvision.utils.make_grid(feature_ouput1)
                #print(out.shape)
                img=feature_imshow(out)
                #print(type(img),img.shape,img*255)
                imgs=img[:,:,0]
                
                imgs=Image.fromarray((imgs*255).astype(np.uint16))
                #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                plt.imshow(imgs,cmap='gray')
                #print(img.shape)
                
                #print(imgs)
                plt.imsave("feature.png",imgs)
                
                """
                #----------------------------------------------
                
                if cfg.mask_proto_prototypes_as_features:#F
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()
                
                if cfg.mask_proto_bias:#F
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)


        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }
            
            if cfg.use_mask_scoring: #F
                pred_outs['score'] = []

            if cfg.use_instance_coeff: #F
                pred_outs['inst'] = []
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers): #[0,1,2,3,4]
                pred_x = outs[idx]#p3,p4,p5,p6,p7
                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features: #1=1&F
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)#插
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)
                
                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]: #T&FTTTT
                    pred_layer.parent = [self.prediction_layers[0]]
                if self.only_last_layer:#T
                    p = pred_layer(pred_x.detach()) #切斷反向傳播
                else:
                    p = pred_layer(pred_x)
                
                for k, v in p.items():
                    pred_outs[k].append(v)
                #print(np.array(pred_outs['loc']).shape[0])
                #for i in pred_outs['loc']:
                #  print(i.shape)    
        for k, v in pred_outs.items():#loc,conf,mask,priors
            pred_outs[k] = torch.cat(v, -2) #全接合4,class14,k32
        #-------------------------------------
        """
        print(pred_outs["mask"])
        print(pred_outs["mask"].shape)
        img=img[:,:,0]
        maskss=pred_outs["mask"].cpu().numpy()
        masks=img @ maskss
        masks = cfg.mask_proto_mask_activation(masks)
        masks=Image.fromarray((masks*255).astype(np.uint16))
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        plt.imshow(masks,cmap='gray')
        #print(img.shape)
                
        #print(imgs)
        plt.imsave("crop.png",masks)
        """
        #---------------------------------------
        if proto_out is not None:#T
            pred_outs['proto'] = proto_out
        
        if self.training:#T
            # For the extra loss functions
            if cfg.use_class_existence_loss:#F
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:#T
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])#1,13,69,69
                
            return pred_outs
        else:
            if cfg.use_mask_scoring:
                pred_outs['score'] = torch.sigmoid(pred_outs['score'])

            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0 ] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            else:

                if cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    
                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
                        * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
                    
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1) #dim最後一維
            #for i,j in pred_outs.items():
            #  print(i,j.shape)
            return self.detect(pred_outs, self)




# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    # Use the first argument to set the config if you want
    import sys
    if len(sys.argv) > 1:
        from data.config import set_cfg
        set_cfg(sys.argv[1])

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # GPU
    net = net.cuda()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

    print()
    for k, a in y.items():
        print(k + ': ', a.size(), torch.sum(a))
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
