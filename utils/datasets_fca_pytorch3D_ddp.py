# Dataset utils and dataloaders

import glob
import hashlib
import logging
import os
import random
import shutil
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xywhn2xyxy, segments2boxes,  xyxy2xywhn
from utils.torch_utils import torch_distributed_zero_first
# import utils.nmr_test as nmr

import os
import sys
import torch
import pytorch3d
import math
import time
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
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
    AmbientLights,
    TexturesVertex
)

# add path for demo utils functions

# Parameters
help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
def draw_red_origin(file_path):
    # 打开图像文件
    image = Image.open(file_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 创建一个新的图像对象，用于绘制点
    new_image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(new_image)

    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 绘制红色的原点（半径为3个像素）
    radius = 1
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=(255, 0, 0))

    # 合并原始图像和绘制的点
    print(new_image.size,image.convert('RGBA').size)
    result_image = Image.alpha_composite(image.convert('RGBA'), new_image)

    # 保存结果图像
    result_file_path = file_path
    result_image.save(result_file_path)

    return result_file_path
def get_params(carlaTcam, carlaTveh):  # carlaTcam: tuple of 2*3
    scale = 0.39
    #scale=0.0592
    #scale = 0.38
    #scale = 0.060467

    # calc eye
    eye = [0, 0, 0]
    for i in range(0, 3):
        # eye[i] = (carlaTcam[0][i] - carlaTveh[0][i]) * scale
        eye[i] = carlaTcam[0][i] * scale

    # calc camera_direction and camera_up
    pitch = math.radians(carlaTcam[1][0])
    yaw = math.radians(carlaTcam[1][1])
    roll = math.radians(carlaTcam[1][2])
    # 需不需要确定下范围？？？
    cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    cam_up = [math.cos(math.pi / 2 + pitch) * math.cos(yaw), math.cos(math.pi / 2 + pitch) * math.sin(yaw),
              math.sin(math.pi / 2 + pitch)]

    # 如果物体也有旋转，则需要调整相机位置和角度，和物体旋转方式一致
    # 先实现最简单的绕Z轴旋转
    p_cam = eye
    p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
    p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
    p_l = [p_cam, p_dir, p_up]
    trans_p = []
    for p in p_l:
        if math.sqrt(p[0] ** 2 + p[1] ** 2) == 0:
            cosfi = 0
            sinfi = 0
        else:
            cosfi = p[0] / math.sqrt(p[0] ** 2 + p[1] ** 2)
            sinfi = p[1] / math.sqrt(p[0] ** 2 + p[1] ** 2)
        cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
        sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
        trans_p.append([math.sqrt(p[0] ** 2 + p[1] ** 2) * cossum, math.sqrt(p[0] ** 2 + p[1] ** 2) * sinsum, p[2]])

    return trans_p[0], \
           [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
           [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def neural_render_to_pytorch3D(cam_trans,veh_trans):
    #todo
    cam_trans_location=cam_trans[0][0:3]
    cam_trans_rotation=cam_trans[1][0:3]
    x=cam_trans_location[0]
    y=cam_trans_location[1]
    z=cam_trans_location[2]
    #print(f"vehicle_trans:{ve}")
    pitch=cam_trans_rotation[0]
    yaw=cam_trans_rotation[1]
    dist=np.sqrt(x**2+y**2+z**2)
    elev=np.arctan(z/np.sqrt(x**2+y**2))
    azim=np.arctan(x/y)
    #打印x,y,z,pitch,yaw,得到的结果是dist,elev,azim
    print(f"x:{x}")
    print(f"y:{y}")
    print(f"z:{z}")
    # print(f"pitch:{pitch}")
    # print(f"yaw:{yaw}")
    # print(f"dist:{dist}")
    # print(f"elev:{elev}")
    # print(f"azim:{azim}")
    return dist,elev,azim

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride,faces, texture_size, verts, aux,texture_img, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix='',mask_dir='', ret_mask=False, phase='training'):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, faces, texture_size, verts,aux,texture_img,imgsz, batch_size,
                                        augment=augment,  # augment images
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=rect,  # rectangular training
                                        cache_images=cache,
                                        single_cls=opt.single_cls,
                                        stride=int(stride),
                                        pad=pad,
                                        image_weights=image_weights,
                                        prefix=prefix, mask_dir=mask_dir, ret_mask=ret_mask, phase=phase)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=rank) if rank != -1 else None #rank=-1就是不需要分布式计算，如果分布式计算就需要使用采样器
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=False,
                        #shuffle=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset,sampler


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def img2label_paths(img_paths, phase='training'):
    # Define label paths as a function of image paths
    if phase == 'training':
        sa, sb = os.sep + 'train_new' + os.sep, os.sep + 'train_label_new' + os.sep  # /images/, /labels/ substrings
    else:
        sa, sb = os.sep + 'test_new' + os.sep, os.sep + 'test_label_new' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, faces, texture_size, verts,aux,texture_img, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='',mask_dir='', ret_mask=False,phase='training'):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.phase = phase
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files, phase)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        # cache.pop('hash')  # remove hash
        # cache.pop('version')  # remove version
        if phase == 'training':
            [cache.pop(k) for k in ('hash', 'version')]  # remove items
            # [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        else:
            [cache.pop(k) for k in ('hash', 'version')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()
        # renderer
        self.device = texture_img.device
        self.faces = faces
        self.verts = verts
        self.verts_uvs = aux.verts_uvs.to(self.device)  # (V, 2)
        self.faces_uvs = faces.textures_idx.to(self.device) 
        tex = TexturesUV(
            verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=texture_img
        )
        self.mesh=Meshes(
            verts=[verts.to(self.device)], faces=[faces.verts_idx.to(self.device)], textures=tex
        )
        raster_settings = RasterizationSettings(
            image_size=640, 
            blur_radius=0.0,  #光栅化过程中不应用模糊。模糊半径为0.0意味着边缘锐利，没有平滑处理。
            faces_per_pixel=1,#光栅化过程中，每个像素只考虑一个面，这适用于不需要多面精度的可视化场景。
            bin_size=0  ,
        )
        R, T = look_at_view_transform(2.7, 0, 180) 
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        lights = AmbientLights(device=self.device)
        self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights
            )
        )
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                logging.info(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            logging.info(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.2  # cache version
        try:
            torch.save(x, path)  # save cache for next time
            logging.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            logging.info(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # path not writeable
        return x

    def set_textures(self, img):
        tex=TexturesUV(
            verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=img
        )
        self.mesh.textures=tex

    def __len__(self):
        return len(self.img_files)

    def set_textures_255(self, textures_255):
        self.textures_255=textures_255

    def __getitem__(self, index):
        img, (h0, w0), (h, w), (veh_trans, cam_trans) = load_image(self, index) #从原始图像就可以获得车辆角度和相机角度
        #camera parameters
        # dist,elev,azim = neural_render_to_pytorch3D(cam_trans, veh_trans) #因为数据集中的车辆不带旋转，只考虑了相机的位置和角度，相机的位置带了权重，可能是考虑了汽车正则化的原因
        # eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
        #R, T = look_at_view_transform(dist, elev, azim)
        cam_trans_location=cam_trans[0][0:3]
        scale=1
        # at=
        cam_trans_rotation=cam_trans[1][0:3]
        # print(f"cam_trans_location:{cam_trans_location}")
        # print(f"cam_trans_rotation:{cam_trans_rotation}")
        
        x=cam_trans_location[0]
        y=cam_trans_location[1]
        z=cam_trans_location[2]
        #time.sleep(10)
        # print(f"x:{x}")
        # print(f"y:{y}")
        # print(f"z:{z}")
        # x,y,z合成eye
        #eye=torch.tensor([0,3,0],dtype=torch.float32).to(self.device).unsqueeze(0)*scale
        
        eye=torch.tensor([x,z,y],dtype=torch.float32).to(self.device).unsqueeze(0)*scale
        #at=((0,0,0.1))
        #double to float
        # eye=torch.tensor(eye).float().to(self.device).unsqueeze(0)
        # camera_up=torch.tensor(camera_up).float().to(self.device).unsqueeze(0)

        # print(f"eye:{eye}")
        R, T = look_at_view_transform(eye=eye)
        # cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T,)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T,fov=90,degrees=True)
        
        # Render the mesh
        imgs_pred = self.renderer(self.mesh, cameras=cameras) #再根据物体的片图结合之前输入的角度和距离信息就可以得到imgs_pred
        imgs_pred=imgs_pred[0, ..., :3]
        imgs_pred=imgs_pred.squeeze(0)
        imgs_pred=imgs_pred.transpose(2, 0).transpose(1, 2)
        # imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures)#再根据物体的片图结合之前输入的角度和距离信息就可以得到imgs_pred
        # Image.fromarray(np.uint8(255 * imgs_pred.squeeze().cpu().data.numpy().transpose(1, 2, 0))).show()
        
        imgs_pred = imgs_pred / torch.max(imgs_pred)
        # print(f"imgs_pred.shape{imgs_pred.shape}")
        # load mask, note that for simplicity, we get the mask via applying segmentation on the rendered image (i.e., imgs_pred)
        if self.ret_mask:
            mask_file = os.path.join(self.mask_dir, "%s.png" % os.path.basename(self.img_files[index])[:-4])
            # print(mask_file)
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
            mask = torch.from_numpy(mask.astype('float32')).to(self.device)
        # Letterbox
        shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        if labels.size:  # normalized xywh to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0])  # xyxy to xywh normalized

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img_cut=( img) * mask
        
        # print(imgs_pred.shape)
        # print(img.shape)
        # print(mask.shape)
        # Applying mask, the transformation function in paper
        # print(f"img.shape{img.shape}")
        # print(f"mask.shape{mask.shape}")
        # print(f"imgs_pred.shape{imgs_pred.shape}")
        img = (1 - mask) * img + (255 * imgs_pred) * mask
        
        imgs_pred=mask*imgs_pred
        
        # return img.squeeze(0), imgs_pred.squeeze(0), mask, imgs_ref.squeeze(0),img_cut.squeeze(0),labels_out, self.img_files[index], shapes
        return img.squeeze(0), imgs_pred.squeeze(0), mask,img_cut.squeeze(0),labels_out, self.img_files[index], shapes



    @staticmethod
    def collate_fn(batch):
        img, texture_img, masks,img_cut, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.stack(texture_img, 0),torch.stack(masks, 0), torch.stack(img_cut, 0),torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


def load_image(self, index):

    """
    Load simulated image and location inforamtion
    """

    # loads 1 image from dataset, returns img, original hw, resized hw
    path = self.img_files[index]
    if self.phase == 'training':
        sa, sb = os.sep + 'train_new' + os.sep, os.sep + 'train' + os.sep  # /images/, /labels/ substrings
    else:
        sa, sb = os.sep + 'test_new' + os.sep, os.sep + 'test' + os.sep  # /images/, /labels/ substrings

    path = sb.join(path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.npz'
    data = np.load(path, allow_pickle=True)  # .item() # .item()      #
    img = data['img']
    # img = img[:, :, ::-1]  # 列表数组左右翻转
    # the relation among veh_trans or cam_trans and img
    veh_trans, cam_trans = data['veh_trans'], data['cam_trans']
    # cam_trans[0][2]-=0.81
    
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                         interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    return img, (h0, w0), img.shape[:2], (veh_trans, cam_trans)  # img, hw_original, hw_resized


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
