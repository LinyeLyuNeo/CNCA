
import argparse
import logging
import os
import time
from pathlib import Path
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.utils.data
import torch.nn as nn
import yaml
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image, ImageDraw
from models.yolo import Model
from utils.datasets_fca_pytorch3D_ddp import create_dataloader
from utils.general_fca  import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
     get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, colorstr
# from utils.general_ddp  import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
#      get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
#     check_requirements, set_logging, colorstr
    
    
from utils.google_utils import attempt_download
from utils.loss_fca_new_ddp import ComputeLoss
# from utils.loss_fca_new_ddp import ComputeLoss

from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
# import neural_renderer
from PIL import Image
from Image_Segmentation.network import U_Net
import sys
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

## add import from diffusion

import copy
import cv2

from functools import partial
from itertools import chain 
from torch import autocast 
from pytorch_lightning import seed_everything
from basicsr.utils import tensor2img 

from ldm.inference_base import (
    DEFAULT_NEGATIVE_PROMPT,
    diffusion_inference,
    diffusion_inference_prompt_optim,
    get_adapters,
    get_sd_models,
)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser
import numpy as np
from PIL import Image

## import for aesthetic predictor 
import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings

import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image, to_tensor





supported_cond = ["style", "sketch", "color", "depth", "canny"]



logger = logging.getLogger(__name__)

# tb_save_dir = 'exp-20240425_exp2_tensorboard_debug_cond_prompt_optim_save_embedding_cosine_clip_sim_clamp_simpson+num_extra_dim-40+λ_sim-1+prompt_tau-0.4+epoch-5+step-20+batch_size-8+lr-0.01+'

writer = SummaryWriter()



# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# RANK = int(os.getenv('RANK', -1))
# WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# GIT_INFO = check_git_info()

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def normalized_torch(a, axis=-1, order=2):
    import torch
    
    
    a_copy = a.clone()
    # a_copy2= a.clone()

    # l2 = torch.norm(a_copy, p=order, dim=axis, keepdim=True)
    # l2[l2 == 0] = 1
    
    
    
    a_norm = torch.norm(a, p=order, dim=axis, keepdim=True)
    
    return a_copy / a_norm

def run_diffusion(args, sd_model, adapters, cond_models, coadapter_fuser, sampler, opt, cond_prompt_optim):

    # print(f"args[-8:]: {args[-8:]}")

    # with torch.inference_mode(), sd_model.ema_scope(), autocast("cuda"):
    with sd_model.ema_scope(), autocast("cuda"):

        inps = []
        for i in range(0, len(args) - 8, len(supported_cond)):
            inps.append(args[i : i + len(supported_cond)])

        opt = copy.deepcopy(opt)

        (
            opt.prompt,
            opt.neg_prompt,
            opt.scale,
            opt.n_samples,
            opt.seed,
            opt.steps,
            opt.resize_short_edge,
            opt.cond_tau,
        ) = args[-8:]

        conds = []
        activated_conds = []
        prev_size = None

      

        for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
            cond_name = supported_cond[idx]

       

            if b == "Nothing":
                if cond_name in adapters:
                    adapters[cond_name]["model"] = adapters[cond_name]["model"].cpu()
            else:
                activated_conds.append(cond_name)
                if cond_name in adapters:
                    adapters[cond_name]["model"] = adapters[cond_name]["model"].to(
                        # opt.diffusion_device
                        opt.device
                    )
                else:
                    adapters[cond_name] = get_adapters(
                        opt, getattr(ExtraCondition, cond_name)
                    )
                adapters[cond_name]["cond_weight"] = cond_weight

                process_cond_module = getattr(api, f"get_cond_{cond_name}")

                if b == "Image":
                    if cond_name not in cond_models:
                        cond_models[cond_name] = get_cond_model(
                            opt, getattr(ExtraCondition, cond_name)
                        )
                    if prev_size is not None:
                        image = cv2.resize(
                            im1, prev_size, interpolation=cv2.INTER_LANCZOS4
                        )
                    else:
                        image = im1
                    conds.append(
                        process_cond_module(opt, image, "image", cond_models[cond_name])
                    )
                    if (
                        idx != 0 and prev_size is None
                    ):  # skip style since we only care spatial cond size
                        h, w = image.shape[:2]
                        prev_size = (w, h)
                else:
                    if prev_size is not None:
                        try:
                            image = cv2.resize(
                                im2, prev_size, interpolation=cv2.INTER_LANCZOS4
                            )
                        except:
                            break
                    else:
                        image = im2
                        
                  
                    conds.append(process_cond_module(opt, image, cond_name, None))
                    
              
                    if (
                        idx != 0 and prev_size is None
                    ):  # skip style since we only care spatial cond size
                        h, w = image.shape[:2]
                        # h, w = cond_optim.shape[:2]
                        prev_size = (w, h)

        features = dict()
        for idx, cond_name in enumerate(
            activated_conds
        ):  # based on the activated_conds, build dict of features

            
            

            cur_feats = adapters[cond_name]["model"](conds[idx])  ## pass the cond_image to the adapter model and obtain cur_feats 
            
            
            
            
            if isinstance(cur_feats, list):  # if cur_feats is a list
                for i in range(
                    len(cur_feats)
                ):  # calcuate the cur_feats with cond_weight
                    cur_feats[i] *= adapters[cond_name]["cond_weight"]
            else:  # calcuate the cur_feats with cond_weight
                cur_feats *= adapters[cond_name]["cond_weight"]
            features[cond_name] = cur_feats  # asign the cur_feats to cond_name features
            
            
        

        adapter_features, append_to_context = coadapter_fuser(
            features
        )  # based on features to create adapter_feastures and append_to_context

        seed_everything(opt.seed)  # adding seed number

        
        # for _ in range(1):  # based on n_samples
        result = diffusion_inference_prompt_optim(
            opt, sd_model, sampler, adapter_features,cond_prompt_optim, append_to_context 
        )  # diffusion inference step
            # ims.append(tensor2img(result, rgb2bgr=False))
        
        

        # Clear GPU memory cache so less likely to OOM
        torch.cuda.empty_cache()
        # return ims, output_conds
        return result
    
def run_diffusion_and_get_output_image(inps, sd_model, adapters, cond_models, coadapter_fuser, sampler, diff_opt, cond_prompt_optim):
    output_tensor = run_diffusion(inps, sd_model, adapters, cond_models, coadapter_fuser, sampler, diff_opt, cond_prompt_optim)
    output_image = F.interpolate(output_tensor, size=(2048, 2048), mode='bilinear', align_corners=False)
    output_image = output_image.permute(0, 2, 3, 1)
    return output_image



def loss_smooth(img, mask, T):

    # [1,3,223,23]
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2) #xi,j − xi+1,j
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2) #xi,j − xi,j+1
    # [3,223,223]
    mask = mask[:, :,:-1, :-1]

    # mask = mask.unsqueeze(1)
    return T * torch.sum(mask * (s1 + s2))


def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, CONTENT=False,):

    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures

def mix_image(image_optim, mask,origin_image):
    return (1 - mask) * origin_image + mask * image_optim


def _convert_image_to_rgb(image):
    return image.convert('RGB')  
    
    

    
# def train(hyp, opt, device):
def train(device,hyp, opt,log_dir,logger,diff_opt):
    
    
    
    
    with torch.backends.cuda.sdp_kernel(enable_flash=False) as disable, torch.autograd.set_detect_anomaly(True):
        logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        T,save_dir, epochs, batch_size, total_batch_size, weights, rank = \
            opt.t,Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.nr*opt.gpus+device
        print(f"rank{rank}")
        torch.manual_seed(100)
        dist.init_process_group(backend='nccl',init_method='env://',world_size=opt.world_size,rank=rank)
        torch.cuda.set_device(device)
        device=torch.device(device)
        
        # ---------------------------------#
        # -----Load aesthetic model--------#
        # ---------------------------------#
        
        aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        # s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        s = torch.load("/home/xxx/Projects/RAUCA/Full-coverage-camouflage-adversarial-attack/src/aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth")

        aesthetic_model.load_state_dict(s)

        aesthetic_model.to(device)
        aesthetic_model.eval()
        
        clip_model, clip_transform = clip.load("ViT-L/14", device=device)  #RN50x64

        # ---------------------------------#
        # ------Load diffusion model-------#
        # ---------------------------------#

        # stable-diffusion model
        # diff_opt.device = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )
        diff_opt.device = device 
        sd_model, sampler = get_sd_models(diff_opt)
        
        sd_model.eval()
        
        
        # adapters and models to processing condition inputs
        adapters = {}
        cond_models = {}

        torch.cuda.empty_cache()
        
        # fuser is indispensable
        coadapter_fuser = CoAdapterFuser(
            unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3
        )  # create co-adapter fuser network
        coadapter_fuser.load_state_dict(
            torch.load(f"t2i_models/coadapter-fuser-sd15v1.pth")
        )  # load pretrain fuser network
        coadapter_fuser = coadapter_fuser.to(diff_opt.device)  # put coadapter_fuser to device  
        
        # initialize the buttons, images, and condition weights
        btns = []
        ims1 = []
        ims2 = []
        cond_weights = []
        
        for cond_name in supported_cond:  # for each supported_cond
            if cond_name == "style":
                btn1 = "Nothing"
                im1 = None
                im2 = None
                cond_weight = 1
            elif cond_name == "sketch":
                # btn1 = cond_name
                # btn1 = "Nothing"
                # btn1 = cond_name
             
                # image = Image.open(image_path)
                # image_array = np.array(image)
                # im1 = image_array
                # im2 = image_array
                # cond_weight = 1
                
                btn1 = "Nothing"
                im1 = None
                im2 = None
                cond_weight = 1
                
            elif cond_name == "color":
                # btn1 = cond_name
          
                # image = Image.open(image_path)
                # image_array = np.array(image)
                # im1 = image_array
                # im2 = image_array
                # cond_weight = 1
                
                btn1 = "Nothing"
                im1 = None
                im2 = None
                cond_weight = 1
            elif cond_name == "depth":
                btn1 = "Nothing"
                im1 = None
                im2 = None
                cond_weight = 1
            elif cond_name == "canny":
                btn1 = "Nothing"
                im1 = None
                im2 = None
                cond_weight = 1
            else:
                btn1 = "Nothing"
                im1 = None
                im2 = None
                cond_weight = 1
            
            btns.append(btn1)
            ims1.append(im1)
            ims2.append(im2)
            cond_weights.append(cond_weight)
        
        prompt = ""

        # neg_prompt = DEFAULT_NEGATIVE_PROMPT
        neg_prompt = ""
        scale = 7.5
        n_samples = 1
        seed = 42
        steps = 20  ## diffusion step
        resize_short_edge = 512
        cond_tau = 1.0
        
        inps = list(
            chain(btns, ims1, ims2, cond_weights)
        )  # concatenate multiple lists to inputs
        inps.extend(
            [prompt, neg_prompt, scale, n_samples, seed, steps, resize_short_edge, cond_tau]
        )
        
        
    
        
        # cond_color= torch.rand(1, 3, 8, 8, dtype=torch.float)
        
        # cond_color_optim = cond_color.to(device).requires_grad_()
        
        ### initilization from pre-defined prompt
        
     
        
        initial_prompt = 'black and white zebra pattern'
        
        
        cond_prompt = sd_model.get_learned_conditioning([initial_prompt])
        
        initial_prompt_param = sd_model.get_learned_conditioning([initial_prompt])
        
        prompt_threshold = opt.prompt_tau
        
        min_value = initial_prompt_param - prompt_threshold
        max_value = initial_prompt_param + prompt_threshold
        
        
        
        
        # ### random initilization of the prompt embedding 
        # cond_prompt = torch.randn(1, 77, 768, dtype=torch.float)

        num_extra_dim = opt.num_extra_dim

        cond_prompt_extra_dim = cond_prompt[:, -num_extra_dim:, :]
        
        
        
        # cond_prompt_optim = cond_prompt.to(device).requires_grad_()

        cond_prompt_extra_dim_optim = cond_prompt_extra_dim.to(device).requires_grad_()
        
 

                
        
        # ---------------------------------#
        # -------Load 3D model-------------#
        # ---------------------------------#
        texture_size = opt.texturesize

        verts, faces, aux = load_obj(opt.obj_file) 
        tex_maps = aux.texture_images
        
        # image = output_image
        image_origin=None
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
            faces_uvs = faces.textures_idx.to(device)  # (F, 3)
            image_origin = list(tex_maps.values())[0].to(device)[None]
            print(f"image_origin.shape:{image_origin.shape}")
            tex = TexturesUV(
                verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image_origin
            )
            
        

        # mesh = Meshes(
        #     verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
        # )
        # mask_image_dir="/home/xxx/Projects/RAUCA/Full-coverage-camouflage-adversarial-attack/src/car_pytorch3d/mask.png"
        mask_image_dir="/home/xxx/Projects/RAUCA/Full-coverage-camouflage-adversarial-attack/src/car_pytorch3d_last/mask.png"
        mask_image = Image.open(mask_image_dir)#.convert("L")
        
        mask_image = (np.transpose(np.array(mask_image)[:,:,:3],(0,1,2))/255).astype('uint8')
        mask_image = torch.from_numpy(mask_image).to(device).unsqueeze(0)
        # print(f"mask_image.shape:{mask_image.shape}")

        # Image.fromarray(np.transpose(mask_image.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
        #                 os.path.join(log_dir, 'mask_image.png')) 
        # image_optim = torch.autograd.Variable(image.to(device), requires_grad=True)
        image_origin = image_origin.clone().detach()
        # print(f"image_optim.shape:{image_optim.shape}")
        # print(f"image_orgin.shape:{image_orgin.shape}")
        # image_optim_in = mix_image(image_optim, mask_image, image_orgin)
        # print(f"image_optim_in.shape:{image_optim_in.shape}")

        
        
        # optim = torch.optim.Adam([image_optim], lr=opt.lr)
        
        
        
        # optim = torch.optim.Adam([cond_color_optim], lr=opt.lr)
        optim = torch.optim.Adam([cond_prompt_extra_dim_optim], lr=opt.lr)
        
        
        
        

        mask_dir = os.path.join(opt.datapath, 'masks_origin_size_640/')

        # ---------------------------------#
        # -------Yolo-v3 setting-----------#
        # ---------------------------------#
        # Directories
        wdir = save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        results_file = save_dir / 'results.txt'

        # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.safe_dump(hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.safe_dump(vars(opt), f, sort_keys=False)

        # Configure
        cuda = device.type != 'cpu'
        init_seeds(2 + rank)
        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)

        # loggers = {'wandb': None}  # loggers dict
        # if rank in [-1, 0]:
        #     opt.hyp = hyp  # add hyperparameters
        #     run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        #     wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        #     loggers['wandb'] = wandb_logger.wandb
        #     data_dict = wandb_logger.data_dict
        #     if wandb_logger.wandb:
        #         weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
       
        nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

        # Model
        pretrained = weights.endswith('.pt')
        with torch_distributed_zero_first(rank):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        #print(f"ckpt['model']:{ckpt['model']}")
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))
        with torch_distributed_zero_first(rank):
            check_dataset(data_dict)  # check
        train_path = data_dict['train']
        test_path = data_dict['val']

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
        logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

        # EMqa
        ema = ModelEMA(model) if rank in [-1, 0] else None

        # Resume
        if pretrained:
            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']
            # Results
            if ckpt.get('training_results') is not None:
                results_file.write_text(ckpt['training_results'])  # write results.txt


        # Image sizes
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)，         
        #det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module

        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
        print(f"rank:{rank}")
        # DP mode
        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model) #

        # ---------------------------------#
        # -------Load dataset-------------#
        # ---------------------------------#
        print(f"train_path:{train_path}")
        
        ###### use diffusio model to create the image_optim （1, H, W, C ）
        # image_optim = run_diffusion_and_get_output_image(inps, sd_model, adapters, cond_models, coadapter_fuser, sampler, diff_opt, cond_color_optim)
        
        
        image_optim = torch.rand(1, 2048, 2048, 3, dtype=torch.float).to(device)
        
        
        image_optim_in = mix_image(image_optim, mask_image, image_origin)
        dataloader, dataset, _ = create_dataloader(train_path, imgsz, batch_size, gs, faces, texture_size, verts, aux,image_optim_in,opt,
                                                hyp=hyp, augment=True, cache=opt.cache_images, rank=rank,
                                                world_size=opt.world_size, workers=opt.workers,
                                                prefix=colorstr('train: '), mask_dir=mask_dir, ret_mask=True)##

        # if cuda and rank != -1:
        #     model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
        #                 # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
        #                 find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
        # ---------------------------------#
        # -------Yolo-v3 setting-----------#
        # ---------------------------------#
        # textures_255_in = cal_texture(texture_255, texture_origin, texture_mask)
        # dataset.set_textures_255(textures_255_in)
        nb = len(dataloader)  # number of batches
        print(f"nb:{nb}")
        # Model parameters
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
        model.names = names

        # Start training
        t0 = time.time()
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        # init loss class
        # ---------------------------------#
        # ------------Training-------------#
        # ---------------------------------#
        model_nsr=U_Net()
        saved_state_dict = torch.load('./NRP_checkpoint/model_nsr_s9_l17.pth')
       


        
        new_state_dict = {}
        for k, v in saved_state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        saved_state_dict = new_state_dict
        model_nsr.load_state_dict(saved_state_dict)
        model_nsr.to(device)

        epoch_start=1+opt.continueFrom
        net = torch.hub.load('yolov3',  'custom','yolov3_9_5.pt',source='local')  
        net.eval()
        net = net.to(device)
        # compute_loss = ComputeLoss(net.model)
        compute_loss = ComputeLoss(model)
        cos_loss = nn.CosineEmbeddingLoss()
        
        

        n_px = 224
        
        
        λ_aes = 0.01
        
        λ_sim = 1

        preprocess_transform = Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            # lambda image: _convert_image_to_rgb(image),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        for epoch in range(epoch_start, epochs+1):  # epoch ------------------------------------------------------------------
            
            model_nsr.eval()

        #     batch = next(iter(dataloader))

   
        #     print(f"batch.shape:{batch.shape}")
            pbar = enumerate(dataloader)
            # print(f"dataloader.dtype:{dataloader.dtype}")
            
            ###### use diffusion model to create the image_optim （1, H, W, C ）
            
            
            
            
            
            image_optim_in = mix_image(image_optim, mask_image, image_origin)
            dataset.set_textures(image_optim_in)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'a_ls', 's_ls','t_loss','labels','tex_mean','grad_mean'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            model.eval()
            #print(dataloader)
            
            mloss = torch.zeros(1, device=device)
            s_mloss=torch.zeros(1)
            a_mloss=torch.zeros(1)
            clip_mloss=torch.zeros(1)
            for i, (imgs, texture_img, masks,imgs_cut, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                
                print(f"batch index: {i}")
                
                # print(imgs.shape)
                # print(texture_img.shape)
                # print(masks.shape)
                # print(imgs_cut.shape)
                # print(targets.shape)
                # uint8 to float32, 0-255 to 0.0-1.0

                #TEST
                imgs_cut = imgs_cut.to(device, non_blocking=True).float() / 255.0
                imgs_in= imgs_cut[0]*masks[0]+imgs[0]*(1-masks[0])/ 255.0 
                # out_tensor = model_nsr(imgs_cut)
                # sig = nn.Sigmoid()
                # out_tensor=sig(out_tensor)  # forward
                # tensor1 = out_tensor[:,0:3, :, :]
                # tensor2 = out_tensor[:,3:6, :, :]
                # # print(tensor1.shape)
                # # print(tensor2.shape)
                
                # tensor3=torch.clamp(texture_img*tensor1+tensor2,max=1)
                out_tensor = model_nsr(imgs_cut)
                sig = nn.Sigmoid()
                out_tensor = sig(out_tensor)
                tensor1 = out_tensor[:,0:3, :, :]
                tensor2 = out_tensor[:,3:6, :, :]
                # tensor3=sig(texture_img*tensor1+tensor2)
                
                tensor3=torch.clamp(texture_img*tensor1+tensor2,0,1)
                masks=masks.unsqueeze(1).repeat(1, 3, 1, 1)
                
                imgs=(1 - masks) * imgs +(255 * tensor3) * masks
                imgs = imgs.to(device, non_blocking=True).float() / 255.0 
                # out, train_out = net.model(imgs)  # forward
                
                out, train_out = model(imgs)  # forward
                #print(f"imgs:{imgs.shape}")
                
                texture_img_np = 255*(imgs.detach()).data.cpu().numpy()[0]
                texture_img_np = Image.fromarray(np.transpose(texture_img_np, (1, 2, 0)).astype('uint8'))
                imgs_show=net(texture_img_np)
                imgs_show.save(log_dir)
                
                texture_img_in_np = 255*(imgs_in.detach()).data.cpu().numpy()
                texture_img_in_np = Image.fromarray(np.transpose(texture_img_in_np, (1, 2, 0)).astype('uint8'))
                imgs_in_show=net(texture_img_in_np)
                imgs_in_show.save(f"{log_dir}/imgs_in")
                
                
                
                # compute loss

                loss1 = compute_loss(out, targets.to(
                    device))

                # print(f"loss_items:{loss_items}")
                # print(f"train_out:{train_out}")
                # print(f"loss_items:{loss_items}")
                
                print(f"Adv loss: {loss1}")
                
            
                loss2 = loss_smooth(tensor3, masks, T)
                
                print(f"Smooth loss: {loss2}")
                
                
                
                
                # image_optim_preprocess = image_encoder_preprocess(image_optim)
                
                image_optim_permuted = image_optim.permute(0, 3, 1, 2)
                
               
                image_optim_preprocess = preprocess_transform(image_optim_permuted)  # preprocess the image
                
                
                
                image_features = clip_model.encode_image(image_optim_preprocess) # encode the image
                
                text_features = clip_model.encode_text(clip.tokenize([initial_prompt]).to(device)) # encode the text

                image_features_copy = image_features.clone()
                text_features_copy = text_features.clone()

                image_features_copy = image_features / image_features.norm(dim=1, keepdim=True)
                text_features_copy = text_features / text_features.norm(dim=1, keepdim=True)
                
                # im_emb_arr = normalized_torch(image_features)
                
                # prediction = aesthetic_model(im_emb_arr.to(device).type(torch.cuda.FloatTensor))
                
                # loss3 = - λ_aes * prediction
                
                # print(f"Aesthetic loss: {loss3}")
                
                # clip_sim = image_features @ text_features.T
                clip_sim = image_features_copy @ text_features_copy.T
                
                print(f"Clip similarity score: {clip_sim}")
                
                # loss5 = - λ_sim * torch.where(clip_sim > 0.25, 20*clip_sim-5, -20*clip_sim)
                
                loss5 = - λ_sim * clip_sim
                
                print(f"Clip similarity loss: {loss5}")
                
               
                
                ones_targets = torch.ones(initial_prompt_param.shape[1])     
                
                cond_prompt_optim_combined = torch.cat((initial_prompt_param[:, :(77-num_extra_dim), :], cond_prompt_extra_dim_optim), dim=1)    
                
                             
                loss4 = cos_loss(cond_prompt_optim_combined.squeeze(0), initial_prompt_param.squeeze(0), ones_targets.to(device))
                
                print(f"Cosine embedding loss: {loss4}")
                

                
                loss = loss1 + loss5
                
                # loss = loss1
               
                print(f"Total loss: {loss}")
                
                
                
                # loss = loss1 + loss2 + loss3
                
                
                
                # Backward
                optim.zero_grad()
                loss.backward(retain_graph=False)
                # loss.backward(retain_graph=True)
                # dist.barrier()
                
                # print(f"i index value: {i}")
                # print(f"cond_prompt_optim before step:{torch.sum(cond_prompt_optim)},device:{device}")

                print(f"cond_prompt_extra_dim_optim before step:{torch.sum(cond_prompt_extra_dim_optim)},device:{device}")
                
                # if cond_color_optim.grad is not None:
                
                #     print(f"cond_color_optim.grad mean: {torch.mean(cond_color_optim.grad)}")
                    
                #     grad = cond_color_optim.grad
                #     grad = grad.contiguous()
                #     # print(f"gradbefore:{torch.sum(grad)},device:{device}")
                #     dist.all_reduce(grad)
                #     # print(f"gradafter:{torch.sum(grad)},device:{device}")
                #     dist.barrier()
                #     cond_color_optim.grad = grad
                #     optim.step()
                    
                #     # cond_color_optim = cond_color_optim.contiguous()
                #     # dist.broadcast(cond_color_optim, src=1)
                    
                # else:
                #     optim.step()
                #     print(f"cond_color_optim.grad is None!")
                
                if cond_prompt_extra_dim_optim.grad is not None:
                
                    print(f"cond_prompt_extra_dim_optim.grad mean: {torch.mean(cond_prompt_extra_dim_optim.grad)}")
                    
                    grad = cond_prompt_extra_dim_optim.grad
                    grad = grad.contiguous()
                    # print(f"gradbefore:{torch.sum(grad)},device:{device}")
                    dist.all_reduce(grad)
                    # print(f"gradafter:{torch.sum(grad)},device:{device}")
                    dist.barrier()
                    cond_prompt_extra_dim_optim.grad = grad
                    optim.step()
                    
                    # cond_color_optim = cond_color_optim.contiguous()
                    # dist.broadcast(cond_color_optim, src=1)
                    
                else:
                    optim.step()
                    print(f"cond_prompt_extra_dim_optim.grad is None!")
                    
                # print(f"cond_color_optim after step:{torch.sum(cond_color_optim)},device:{device}")
                


                
                
                print(f"cond_prompt_extra_dim_optim after step:{torch.sum(cond_prompt_extra_dim_optim)},device:{device}")
                
                # upsampled_cond_color_optim = F.interpolate(cond_color_optim, size=(512, 512), mode='nearest')
                
                
             
                if rank == 0:
                    try:

                        Image.fromarray(np.transpose(255 * imgs.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, 'test_total.png')) 
                        
                        Image.fromarray(
                            (255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, 'render_image.png'))
                        # print(f"mask_shape:{masks.shape}")
                        Image.fromarray(np.transpose(255 * masks.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, 'mask.png'))
                        #Image.fromarray(
                        #     (255 * imgs_ref).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        #     os.path.join(log_dir, 'texture_ref.png'))
                        Image.fromarray(
                            (255 * imgs_cut).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, 'img_cut.png'))
                        Image.fromarray(
                            (255 * image_optim).data.cpu().numpy()[0].astype('uint8')).save(
                            os.path.join(log_dir, 'uv_map.png'))
                            
                        
                        
                        if i % 20 == 0:
                            Image.fromarray(
                            (255 * image_optim).data.cpu().numpy()[0].astype('uint8')).save(
                            os.path.join(log_dir, f'uv_map_{epoch}_{i}.png'))
                            
                            
                            
                        # Image.fromarray(
                        #     (255 * cond_color_optim).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        #     os.path.join(log_dir, 'cond_color_optim.png'))
                        # Image.fromarray(
                        # (255 * upsampled_cond_color_optim).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        # os.path.join(log_dir, 'upsampled_cond_color_optim.png'))
                        # Image.fromarray(
                        #     (255 * tensor3).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        #     os.path.join(log_dir, 'img_tensor3.png'))
                        Image.fromarray(
                            (255 * tensor1).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, 'img_tensor1.png'))
                        Image.fromarray(
                            (255 * tensor2).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, 'img_tensor2.png'))
                        Image.fromarray(np.transpose(255*imgs_in.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                                    os.path.join(log_dir, 'input_origin.png'))
                    except:
                        pass

                if rank in [-1, 0]: 
                    
                    mloss = (mloss * i + loss1.detach()) / (i + 1)
                    s_mloss=(s_mloss*i+loss2.detach().data.cpu().numpy()/batch_size) / (i+1)
                    clip_mloss=(clip_mloss*i+loss5.detach().data.cpu().numpy()/batch_size) / (i+1)
                    a_mloss=(a_mloss*i+loss.detach().data.cpu().numpy()/batch_size) / (i+1)
                    
                    
                    if i % 2 == 0:
                    
                        # tb_writer.add_scalar("mean_adv_loss", mloss[0], i)
                        # tb_writer.add_scalar("mean_sm_loss", s_mloss, i)
                        # tb_writer.add_scalar("mean_adv_sm_Loss",a_mloss, i)
                        
                        loss_index_num = nb*(epoch - 1) + i
                        
                        writer.add_scalar("mean_adv_loss/train", mloss[0], loss_index_num)
                        writer.add_scalar("mean_sm_loss/train", s_mloss, loss_index_num)
                        writer.add_scalar("mean_clip_loss/train", clip_mloss, loss_index_num)
                        writer.add_scalar("total_Loss/train",a_mloss, loss_index_num)

                #update texture_param
                # textures = cal_texture(texture_param, texture_origin, texture_mask)
                
                
                ##### use diffusion model to create the image_optim （1, H, W, C ）
                
                # upsampled_cond_color_optim = F.interpolate(cond_color_optim, size=(512, 512), mode='nearest')
                
                # param_sum = sum(p.data.sum() for p in sd_model.parameters())
                # print(f"sum of sd_model parameters: {param_sum}")
                
                # cond_prompt_optim = clipping(cond_prompt_optim, initial_prompt_param, prompt_threshold)
                
                
                
                
                

                # cond_prompt_optim_combined = torch.cat((cond_prompt[:, :(77-num_extra_dim), :], cond_prompt_extra_dim_optim), dim=1)
                
                
                # print(f"cond_prompt_extra_dim_optimbefore clipping {torch.sum(cond_prompt_extra_dim_optim)},device:{device}")
                
                print(f"cond_prompt_extra_dim_optimbefore clipping {torch.sum(cond_prompt_optim_combined)},device:{device}")
                
                cond_prompt_optim_clamped = torch.clamp(cond_prompt_optim_combined, min_value, max_value)
                
                print(f"cond_prompt_extra_dim_optim after clipping {torch.sum(cond_prompt_extra_dim_optim)},device:{device}")
                
                # print(f"cond_prompt_extra_dim_optim after clipping {torch.sum(cond_prompt_optim_clamped)},device:{device}")
                
                 

                  


                
                
                image_optim = run_diffusion_and_get_output_image(inps, sd_model, adapters, cond_models, coadapter_fuser, sampler, diff_opt, cond_prompt_optim_clamped)
                
                
                
                
                dist.barrier()
                
                
                image_optim_in = mix_image(image_optim, mask_image, image_origin)

                # print(f"mask_image_max:{mask_image.max()}")
                dataset.set_textures(image_optim_in)
                
                # break
                
                
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training
            # tb_writer.add_scalar("meanTLoss", mloss[0], epoch)
            # tb_writer.add_scalar("meanSLoss", s_mloss, epoch)
            # tb_writer.add_scalar("AllSLoss",a_mloss, epoch)
            if epoch % 1 == 0:
                np.save(os.path.join(log_dir, f'texture_{epoch}.npy'), image_optim.data.cpu().numpy())
                np.save(os.path.join(log_dir, f'cond_prompt_optim_combined_{epoch}.npy'), cond_prompt_optim_combined.data.cpu().numpy())
                
                
                # Image.fromarray((image_optim.data.cpu().numpy() * 255).astype('uint8')).save(os.path.join(log_dir, f'texture_{epoch}.png'))
        np.save(os.path.join(log_dir, 'texture.npy'), image_optim.data.cpu().numpy())
        # Image.fromarray((image_optim.data.cpu().numpy() * 255).astype('uint8')).save(os.path.join(log_dir, f'texture.png'))

        writer.flush()
        writer.close()
        torch.cuda.empty_cache()
        return results

log_dir = ""
def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
        
    log_folder_name = dir_name
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name
    
    return log_folder_name



if __name__ == '__main__':
    print(f"logger{logger}")
    parser = argparse.ArgumentParser()
    # hyperparameter for training adversarial camouflage
    # ------------------------------------#
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/carla.yaml', help='data.yaml path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for texture_param')
    parser.add_argument('--obj_file', type=str, default='/home/xxx/Projects/RAUCA/Full-coverage-camouflage-adversarial-attack/src/car_pytorch3d/pytorch3d_Etron.obj', help='3d car model obj')
    parser.add_argument('--faces', type=str, default='car_assets/exterior_face.txt',
                        help='exterior_face file  (exterior_face, all_faces)')
    # parser.add_argument('--datapath', type=str, default='/data/xxx/phy_multi_weather_new_day_right',
    #                     help='data path')
    parser.add_argument('--datapath', type=str, default='/data1/xxx/phy_multi_weather_new_day_right',
                            help='data path')
    parser.add_argument('--patchInitial', type=str, default='random',
                        help='data path')
    # parser.add_argument('--device', default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='0,3,4,5,6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='4,5,6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--lamb", type=float, default=1e-4) #lambda
    parser.add_argument("--d1", type=float, default=0.9)
    parser.add_argument("--d2", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=0.0001)
    # parser.add_argument("--prompt_tau", type=float, default=0.1)
    # parser.add_argument("--prompt_tau", type=float, default=0.25)
    # parser.add_argument("--prompt_tau", type=float, default=0.5)
    parser.add_argument("--prompt_tau", type=float, default=1)
    # parser.add_argument("--prompt_tau", type=float, default=0.75)
    # parser.add_argument("--prompt_tau", type=float, default=2)
    parser.add_argument("--num_extra_dim", type=int, default=40)
    parser.add_argument('--epochs', type=int, default=2)
    
    # ------------------------------------#

    #add
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    # parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--classes', nargs='+', type=int, default=[2],
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--continueFrom', type=int, default=0, help='continue from which epoch')
    parser.add_argument('--texturesize', type=int, default=6, help='continue from which epoch')
    parser.add_argument('--nodes',type=int,default=1)
    parser.add_argument('--gpus',type=int,default=4,help="num gpus per node")
    parser.add_argument('--nr',type=int,default=0,help="ranking within the nodes")
    ## diffusion add
    
   
    opt = parser.parse_args()
    
    diff_parser = argparse.ArgumentParser()
    
    diff_parser.add_argument(
    "--sd_ckpt",
    type=str,
    default="t2i_models/v1-5-pruned-emaonly.ckpt",
    help="path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported",
    )
    diff_parser.add_argument(
        "--vae_ckpt",
        type=str,
        default=None,
        help="vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded",
    )
    diff_opt = diff_parser.parse_args()
    
    diff_opt.config = "configs/stable-diffusion/sd-v1-inference.yaml"  # stable-diffusion model config path
    
    for cond_name in supported_cond:
        setattr(
            diff_opt,
            f"{cond_name}_adapter_ckpt",
            f"t2i_models/coadapter-{cond_name}-sd15v1.pth",
        )  # setup adapter model path
        

    # diff_opt.max_resolution = 512 * 704  # default output image resolution
    diff_opt.max_resolution = 512 * 512  # default output image resolution
    diff_opt.sampler = "ddim"  # default sampler method
    diff_opt.cond_weight = 1.0  # default cond_weight
    diff_opt.C = 4  #
    diff_opt.f = 8  #
    # TODO: expose style_cond_tau to users
    diff_opt.style_cond_tau = 1.0
    
   

    
    
    T = opt.t 
    D1 = opt.d1
    D2 = opt.d2
    lamb = opt.lamb
    LR = opt.lr
    Dataset=opt.datapath.split('/')[-1]
    PatchInitial=opt.patchInitial
    logs = {
        'exp': '20240520_exp7_cond_prompt_optim_extra_dim_clamp_clip_zebra',
        'num_extra_dim': opt.num_extra_dim,
        'λ_sim' : 1,
        'prompt_tau': opt.prompt_tau,
        'epoch': opt.epochs,
        'step':'20',
        # 'withNewNSR':"True",
    #    'objall':"True",
        # 'loss':"FCA_sigmoid_class",
        # 'texturesize':opt.texturesize,
        # 'weights':opt.weights,
        # 'dataset':Dataset,
        # 'smooth':"tensor3",
        # 'patchInitialWay':PatchInitial,
        'batch_size': opt.batch_size,
        'lr': opt.lr,
        # 'lamb': lamb,
        # 'D1': D1,
        # 'D2': D2,
        # 'T': T, 
    }
    opt.name = make_log_dir(logs)
    print(logs)
    # opt.name = logs
    texture_dir_name = ''
    texture_dir_name = ''
    for key, value in logs.items():
        texture_dir_name+= f"{key}-{str(value)}+"
    
    # Set DDP variables
    

    #opt.world_size = 4 # os.environ[""]
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    print('WORLD_SIZE' in os.environ)

    



    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run   ``
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))
    opt.total_batch_size=opt.batch_size
    #device = select_device(opt.device, batch_size=opt.batch_size)
    # print(f"device:{device}")
    # if opt.local_rank != -1:
    #     msg = 'is not compatible with YOLOv3 Multi-GPU DDP training'
    #     assert not opt.image_weights, f'--image-weights {msg}'
    #     assert not opt.evolve, f'--evolve {msg}'
    #     assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
    #     assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
    #     assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
    #     torch.cuda.set_device(LOCAL_RANK)
    #     device = torch.device('cuda', LOCAL_RANK)
    #     dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    # # Hyperparameters
    # with open(opt.hyp) as f:
    #     hyp = yaml.safe_load(f)  # load hyps
    # # Train
    # logger.info(opt)
    
    global tb_writer
    
    tb_writer = None  # init loggers

    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6007/")
        # tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    # train(hyp, opt, device)
    opt.world_size=opt.nodes*opt.gpus
    opt.lr=opt.lr*opt.world_size
    opt.batch_size = opt.total_batch_size // opt.world_size
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    # Train
    logger.info(opt)
    
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6006'
    os.environ["CUDA_VISIBLE_DEVICES"] =opt.device
    mp.spawn(train,nprocs=opt.gpus,args=(hyp,opt,log_dir,logger,diff_opt),join=True)






