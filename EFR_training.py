import argparse
import logging
import os
import time
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw
from models.yolo import Model
from utils.datasets_NSR_pytorch3D import create_dataloader
from utils.general_NSR import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
     get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, colorstr
from utils.google_utils import attempt_download
from utils.loss_NSR import ComputeLoss
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

from PIL import Image
from Image_Segmentation.network import U_Net
import torch.multiprocessing as mp
import torch.distributed as dist
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
from plot_image_grid import image_grid

import torch.multiprocessing as mp
import torch.distributed as dist
from itertools import chain
logger = logging.getLogger(__name__)
logger = None

def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, CONTENT=False,):
    # 计算纹理
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)#
    return texture_origin * (1 - texture_mask) + texture_mask * textures
def mix_image(image_optim, mask,origin_image):
    return (1 - mask) * origin_image + mask * image_optim
def calculate_inverse_ratio(masks):

    nonzero_counts = torch.sum(masks != 0, dim=(1, 2), dtype=torch.float)


    total_pixels = masks.size(1) * masks.size(2)


    inverse_ratios = total_pixels / nonzero_counts

    return inverse_ratios

def train(device,hyp, opt,log_dir,logger):
    
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.nr*opt.gpus+device
    torch.manual_seed(100)
    dist.init_process_group(backend='nccl',init_method='env://',world_size=opt.world_size,rank=rank)
    torch.cuda.set_device(device)
    device=torch.device(device)
    print(f"device:{torch.cuda.current_device()}")

    tb_writer = None  # init loggers
    if rank in [-1, 0]:
        print("为什么看不到")
        prefix = colorstr('tensorboard: ')
        text=f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/"
        print(text)
        tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    # ---------------------------------#
    # -------Load 3D model-------------#
    # ---------------------------------#
    texture_size = 6
    verts, faces, aux = load_obj(opt.obj_file) 
    tex_maps = aux.texture_images
    image=None
    if tex_maps is not None and len(tex_maps) > 0:
        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = faces.textures_idx.to(device)  # (F, 3)
        image = list(tex_maps.values())[0].to(device)[None]
        #print(f"image.shape:{image.shape}")
    #     tex = TexturesUV(
    #         verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
    #     )

    # mesh = Meshes(
    #     verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
    # )
    mask_image_dir="/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/car_pytorch3d/mask.png"
    mask_image = Image.open(mask_image_dir)#.convert("L")
    
    mask_image = (np.transpose(np.array(mask_image)[:,:,:3],(0,1,2))/255).astype('uint8')
    mask_image = torch.from_numpy(mask_image).to(device).unsqueeze(0)
    # print(f"mask_image.shape:{mask_image.shape}")

    # Image.fromarray(np.transpose(mask_image.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
    #                 os.path.join(log_dir, 'mask_image.png')) 
    #image_optim=None
    # if rank in [-1, 0]:
    image_optim = torch.autograd.Variable(image.to(device), requires_grad=False) #把这个变量自动优化
    # dist.broadcast(image_optim, src=0)
    # dist.barrier()
    #image_optim = nn.parallel.DistributedDataParallel(image_optim,device_ids=[device],output_device=device)
    image_orgin = image_optim.clone()
    # print(f"image_optim.shape:{image_optim.shape}")
    # print(f"image_orgin.shape:{image_orgin.shape}")
    # image_optim_in = mix_image(image_optim, mask_image, image_orgin)
    # print(f"image_optim_in.shape:{image_optim_in.shape}")

    

    mask_dir = os.path.join(opt.datapath, 'masks/')


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

    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
    #导入数据集的一些参数
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
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude key
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
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
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)，          det = model.module.model[-1] if is_parallel(model) else model.model[-1]  #

    nl = model.model[-1].nl
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    print(f"rank:{rank}")
    # ---------------------------------#
    # -------Load dataset-------------#
    # ---------------------------------#
    print(f"train_path:{train_path}")
    print(f"opt.cache_images:{opt.cache_images}")
    image_optim_in = mix_image(image_optim, mask_image, image_orgin)
    dataloader, dataset,sampler = create_dataloader(train_path, imgsz, batch_size, gs, faces, texture_size, verts, aux,image_optim_in, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            prefix=colorstr('train: '), mask_dir=mask_dir, ret_mask=True)#
    # ---------------------------------#
    # -------Yolo-v3 setting-----------#
    # ---------------------------------#
    #dataset.start()
    
    
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


    model_nsr=U_Net()

    saved_state_dict = torch.load('/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/logs/epoch-9+dataset-DTN+ratio-false+night-4+day-0+patchInitialWay-random+batch_size-16+name-pytorch3d_NRP+pytorch-true+/model_nsr_s9_l10.pth')  # 原始的参数字典
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        name = k[7:]  # 去掉 'module.' 前缀
        new_state_dict[name] = v
    saved_state_dict = new_state_dict
    model_nsr.load_state_dict(saved_state_dict)

    model_nsr.to(device)
    model_nsr = nn.SyncBatchNorm.convert_sync_batchnorm(model_nsr)
    # model_nsr.to(device)
    # model_nsr.cuda(device)
    model_nsr=nn.parallel.DistributedDataParallel(model_nsr,device_ids=[device],output_device=device)
    model_nsr.train()
    # if rank in [-1, 0]:
    #     init_img =torch.zeros((1,3,640,640),device=device)
    #     tb_writer.add_graph(model_nsr, init_img)
    optimizer = optim.Adam(model_nsr.parameters(), lr=0.001)
    optimizer.zero_grad()
    # ---------------------------------#
    # ------------Training-------------#
    # ---------------------------------#
    epoch_start=1+opt.continueFrom
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    #color_list=[[-1,-1,-1],[1,-1,-1],[-1,1,-1],[-1,-1,1],[1,1,-1],[-1,1,1],[1,-1,1],[1,1,1],[1/255,1/255,1/255]]
    color_list=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1],[128/255,128/255,128/255]]

    for epoch_real in range(11,20):
        sampler.set_epoch(epoch_real)
        mloss = torch.zeros(1, device=device)
        for epoch in range(epoch_start, epochs+1):
            # texture_param=(np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3))*np.array(color_list[epoch-1])).astype('float32')
            # texture_param = torch.autograd.Variable(torch.from_numpy(texture_param).to(device), requires_grad=False)
            # textures = cal_texture(texture_param, texture_origin, texture_mask)
            # dataset.set_textures(textures)
            #使image_optim全部为1
            image_optim=(image_optim.fill_(1.0)*torch.from_numpy(np.array(color_list[epoch-1]).astype('float32')).to(device))
            # print(f"image_optim_max:{torch.max(image_optim)}")
            # print(f"image_optim_min:{torch.min(image_optim)}")
            image_optim_in = mix_image(image_optim, mask_image, image_orgin)
            dataset.set_textures(image_optim_in)
            color_0_1 = color_list[epoch-1]
            r = int(color_0_1[0]  * 255)
            g = int(color_0_1[1]  * 255)
            b = int(color_0_1[2]  * 255)
            color_name = f"{r},{g},{b}"
        
            if epoch_real==-1:
                model_nsr.eval()
            else:
                model_nsr.train()
            pbar = enumerate(dataloader)
            # textures = cal_texture(texture_param, texture_origin, texture_mask)
            # dataset.set_textures(textures)
            # logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'loss','labels','tex_mean','grad_mean'))
            logger.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem','mloss','loss'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar

            #print(dataloader)
            
            
            record_start_all=time.perf_counter()
            dataset.set_color(color_name)
            for i, (imgs, texture_img, masks,imgs_cut,imgs_NSR_ref, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                #print(masks.shape)
                
                start_all=time.perf_counter()
                time_all=start_all-record_start_all
                record_start_all=start_all
                
                #将训练图像传递给神经网络
                start_Unet_forward =time.perf_counter()
                imgs_cut = imgs_cut.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
                # print(imgs_cut.shape)
                # print(masks.shape)
                # print(imgs.shape)
                imgs_in= imgs_cut[0]*masks[0]+imgs[0]*(1-masks[0])/ 255.0 
                
                  # forward
                #compute loss
                end_Unet_forward = time.perf_counter()
                time_Unet_forward=end_Unet_forward-start_Unet_forward
                start_Unet_backward=time.perf_counter()
                sig = nn.Sigmoid()
                relu= nn.ReLU()
                # out_tensor=sig(out_tensor)
                out_tensor = model_nsr(imgs_cut)
                sig = nn.Sigmoid()
                out_tensor =sig(out_tensor)
                tensor1 = out_tensor[:,0:3, :, :]
                tensor2 = out_tensor[:,3:6, :, :]
                tensor3=torch.clamp(texture_img*tensor1+tensor2,0,1)
                # tensor1 = out_tensor[:,0:3, :, :]
            
                # tensor3=(texture_img+tensor1)/2.0
                loss=criterion(tensor3,imgs_NSR_ref)
                

                #ratio
                loss_array= torch.zeros(tensor3.shape[0]).to(device)
                for j in range(tensor3.shape[0]):
                    loss_img=criterion(tensor3[j],imgs_NSR_ref[j])
                    loss_array[j]=loss_img
                ratio = calculate_inverse_ratio(masks)
                loss=torch.sum(loss_array*ratio)
                
                
                output=tensor3[0]*masks[0]+imgs[0]/ 255.0 *(1-masks[0])
                output_ref=imgs_NSR_ref[0]*masks[0]+imgs[0]/ 255.0 *(1-masks[0])
                # Backward
                optimizer.zero_grad()
                loss.backward(retain_graph=False) #retain_graph=True
                

                optimizer.step()    
                end_Unet_backward=time.perf_counter()
                time_Unet_backward=end_Unet_backward-start_Unet_backward
                
                                             
                try:
                    if rank in [-1, 0]: 

                        
                        
                        # Image.fromarray(
                        #     (255 * tensor1).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        #     os.path.join(log_dir, 'img_tensor1.png'))
                        # Image.fromarray(
                        #     (255 * tensor2).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                        #     os.path.join(log_dir, 'img_tensor2.png'))
                    #     Image.fromarray(
                    #         (255 * tensor3).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                    #         os.path.join(log_dir, 'result.png'))
                    #     Image.fromarray(
                    #         (255 * imgs_NSR_ref).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                    #         os.path.join(log_dir, 'target.png'))
                    #     Image.fromarray(
                    #             (255*imgs_cut).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                    #             os.path.join(log_dir, "训练图像切割图像.png"))
                        Image.fromarray(
                            (255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, '渲染车.png'))
                        Image.fromarray(np.transpose(255*imgs_in.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, '输入原图.png')) 

                        Image.fromarray(np.transpose(255*output.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, '输出原图.png')) 
                        
                        Image.fromarray(np.transpose(255*output_ref.data.cpu().numpy(), (1, 2, 0)).astype('uint8')).save(
                            os.path.join(log_dir, '输出ref.png'))
                    
                except:
                    pass
                
                
                start_NSR_all=time.perf_counter()
                
                color_index=(i%9+epoch-1)%9
                
                image_optim=(image_optim.fill_(1.0)*torch.from_numpy(np.array(color_list[color_index]).astype('float32')).to(device))
                # print(f"image_optim_max:{torch.max(image_optim)}")
                # print(f"image_optim_min:{torch.min(image_optim)}")
                image_optim_in = mix_image(image_optim, mask_image, image_orgin)
                dataset.set_textures(image_optim_in)
                color_0_1 = color_list[color_index]
                r = int(color_0_1[0]  * 255)
                g = int(color_0_1[1] * 255)
                b = int(color_0_1[2] * 255)
                color_name = f"{r},{g},{b}"
                dataset.set_color(color_name)
                #print(f"i:{i} color:{color_name}")
                
                if rank in [-1, 0]: 
                    
                    mloss = (mloss * i + loss) / (i + 1)
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 +"%10.4f"*2)  % (
                        '%g/%g' % (epoch, epochs), mem,mloss.data,loss.data)

                    pbar.set_description(s)
                    # tb_writer.add_histogram(tag="Conv_last",
                    #             values=model_nsr.model.conv1.weight,
                    #             global_step=epoch_real)
                end_NSR_all=time.perf_counter() 
                end_all=time.perf_counter() 
                #print(f"time_without_load:{end_all-start_all},time_Unet_forward:{time_Unet_forward},time_Unet_backward:{time_Unet_backward}\n,timeall:{time_all},NSR_all:{end_NSR_all-start_NSR_all}")            
               
            # if rank in [-1, 0]: 
            #    tb_writer.add_scalar("BCEmeanloss_pytorch3d", mloss.data, epoch)
                

            # end epoch ----------------------------------------------------------------------------------------------------
        # end training
        if rank in [-1, 0]: 
            torch.save(model_nsr.state_dict(), (os.path.join(log_dir, f'model_nsr_s{epoch}_l{epoch_real}.pth')))
            tb_writer.add_scalar("BCEmeanloss_pytorch3d", mloss.data, epoch_real)
            
    if rank in [-1, 0]: 
        torch.save(model_nsr.state_dict(), 'model_nsr.pth')
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results

log_dir = ""
def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name



if __name__ == '__main__':
    logger= logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    # hyperparameter for training adversarial camouflage
    # ------------------------------------#
    parser.add_argument('--weights', type=str, default='yolov3.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/nsr.yaml', help='data.yaml path')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for texture_param')
    parser.add_argument('--obj_file', type=str, default='/home/zhoujw/FCA/Full-coverage-camouflage-adversarial-attack/src/car_pytorch3d/pytorch3d_Etron.obj', help='3d car model obj')
    parser.add_argument('--faces', type=str, default='car_assets/exterior_face.txt',
                        help='exterior_face file  (exterior_face, all_faces)')
    parser.add_argument('--datapath', type=str, default='/data/zhoujw/DTN',
                        help='data path')
    parser.add_argument('--patchInitial', type=str, default='random',
                        help='data path')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--lamb", type=float, default=1e-4) #lambda
    parser.add_argument("--d1", type=float, default=0.9)
    parser.add_argument("--d2", type=float, default=0.1)
    parser.add_argument("--t", type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=9)
    
    # ------------------------------------#

    #add
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
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
    parser.add_argument('--name', default='pytorch3d_NRP', help='save to project/name')
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
    parser.add_argument('--nodes',type=int,default=1)
    parser.add_argument('--gpus',type=int,default=4,help="num gpus per node")
    parser.add_argument('--nr',type=int,default=0,help="ranking within the nodes")


    opt = parser.parse_args()

    T = opt.t
    D1 = opt.d1
    D2 = opt.d2
    lamb = opt.lamb
    LR = opt.lr
    Dataset=opt.datapath.split('/')[-1]
    PatchInitial=opt.patchInitial
    logs = {
        'epoch': opt.epochs,
        'dataset':Dataset,
        'ratio' : 'false',
        'night' : '4',
        'day' : '0',
        'patchInitialWay':PatchInitial,
        'batch_size': opt.batch_size,
        'name' : opt.name,
        'pytorch': 'true',
    }

    make_log_dir(logs)
    print(logs)
    texture_dir_name = ''
    for key, value in logs.items():
        texture_dir_name+= f"{key}-{str(value)}+"
    
    # Set DDP variables


    
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # set_logging(opt.global_rank)
    # print(f"rank:{opt.global_rank}")
    # if opt.global_rank in [-1, 0]:
    #     check_git_status()
    #     check_requirements(exclude=('pycocotools', 'thop'))

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

    opt.total_batch_size = opt.batch_size
    # device = select_device(opt.device, batch_size=opt.batch_size)

    # #add
    # if opt.local_rank != -1:
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
    #     opt.batch_size = opt.total_batch_size // opt.world_size

    opt.world_size=opt.nodes*opt.gpus
    opt.lr=opt.lr*opt.world_size
    opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    # Train
    logger.info(opt)
    
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    os.environ["CUDA_VISIBLE_DEVICES"] =opt.device
    mp.spawn(train,nprocs=opt.gpus,args=(hyp,opt,log_dir,logger,),join=True)
    



