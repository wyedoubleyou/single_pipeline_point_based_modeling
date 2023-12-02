
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import cv2
import yaml
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path 
from compare import first_px_coor, centering_object
from iou import mov_segm, iou_calc

sys.path.append(os.path.abspath('')+'/src')

from dataloaders.common.bbox import get_square_bbox, scale_bbox
from dataloaders.loaders.utils import make_dataloader

from models.nstack import nstack_from_statedict, PointNeuralTex, NeuralStack
from models.pcd_converter import PCDConverter
from models.pcd_renderer import Renderer

from appearance.infer import infer_pid
from appearance.utils import to_img, show_axes

from utils.utils import load_module
from utils.common import dict2device, itt, tti, to_sigm

def inference_process(rgb_fname): 

    os.environ["DEBUG"] = ''

    # set the subject to take the outfit style from
    style_pid = 'male-3-casual'
    experiment_name = '05-21_21-48___config_name^appearance+psp_male-3-casual'
    experiment_path = f'out/appearance/psp/{style_pid}/{experiment_name}'
    ep = 79

    # appearance model checkpoints setup
    model_path = os.path.join(experiment_path, 'checkpoints', f'model0_{ep:04d}.pth')
    ntex_path = os.path.join(experiment_path, 'checkpoints', f'ntex0_{ep:04d}.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(experiment_path, f'model0_{ep:04d}.pth')
        ntex_path = os.path.join(experiment_path, f'ntex0_{ep:04d}.pth')

    model_cp = torch.load(model_path) #error over here
    ntex_cp = torch.load(ntex_path)
    args = model_cp['args']
    device = 'cuda:0'

    args.num_workers = 0

    # point cloud converter
    converter = PCDConverter(device)

    # outfit point cloud prediction network
    draping_network = load_module('models', 'draping_network_wrapper').Wrapper.get_net(args)
    print(draping_network.glo_stack)

    # learned outfit codes
    outfit_codes_file = f'out/outfit_code/outfit_codes_psp.pkl'
    outfit_codes_dict = pickle.load(open(outfit_codes_file, 'rb'))
    print('\n> Outfit codes dict:', outfit_codes_dict)

    outfit_code = torch.from_numpy(outfit_codes_dict[style_pid]).to(device)
    print(f'\n> Current style: pid={style_pid}, shape={outfit_code.shape}')

    # learned per-point neural descriptors
    ndesc_stack = nstack_from_statedict(lambda: PointNeuralTex(args.ntex_channels, args.pcd_size), ntex_cp)
    print(ndesc_stack)

    # differentiable rasterizer
    args.visibility_thr= 2e-3
    renderer = Renderer(args.imsize, args.imsize, args.ntex_channels, device=args.device, 
                        visibility_thr=args.visibility_thr, scale=1, radius=2)
    renderer_flood = Renderer(args.imsize, args.imsize, args.ntex_channels, device=args.device, 
                            visibility_thr=args.visibility_thr, scale=1, radius=6)

    # neural rendering network
    generator = load_module('generators', args.generator).Wrapper.get_net(args)
    generator.load_state_dict(model_cp['generator'])

    # set the subject to try the learned outfit on
    pose_pid = rgb_fname                                # MODIFY 
    smpid = 1                                           # MODIFY 

    if smpid ==1: 
        smpl_type = 'smpl'
    elif smpid ==2: 
        smpl_type = 'smplx'

    data_root = 'samples/internet_images'
    data_name = data_root.split('/')[-1] # internet_images
    rgb_dir = 'images'                                    # MODIFY 
    segm_dir = 'segmentations'                      # MODIFY 
    smpl_dir = 'smpl'                       # MODIFY 
    splits_dir = None

    datalists = [[pose_pid]] #[['itw_2']] 


    dataloader = make_dataloader(data_name, data_root, rgb_dir, segm_dir, smpl_dir, datalists[0], 
                                smpl_model_path='data/smpl_models/SMPL_NEUTRAL.pkl',
                                train=False, batch_size=1)
    # print(type(dataloader))
    # print(dataloader) # <torch.utils.data.dataloader.DataLoader object at 0x7fc696e5eb20>
        
    for i, (data_dict, target_dict) in enumerate(dataloader):
        # print(i) # i=0(image) 
        data_dict = dict2device(data_dict, device)
        data_dict['cam'] = ''
        data_dict['seq'] = [style_pid] #['male-3-casual']
        # print(len(target_dict))#data_dict: 13 ; 
        # for i in target_dict:
        #     print(i,end=':')
        #     print(target_dict[f'{i}'])
    
        out_dict = infer_pid(style_pid, data_dict, target_dict, outfit_code, ndesc_stack, converter, 
                            draping_network, renderer, renderer_flood, generator, device=device, args=args)
        

        rgb_np, raster_features_np, real_rgb_np, real_segm_np, real_rgb_nos_np, vton_pcd_np, vton_rgb_np = to_img([
            out_dict['rgb_np'], out_dict['raster_features_np'], out_dict['real_rgb_np'], out_dict['real_segm_np'], 
            out_dict['real_rgb_nos_np'], out_dict['vton_pcd_np'], out_dict['vton_rgb_np']])
        
        
        #note: cropp img affected: real_rgb_np, real_segm_np, real_rgb_nos_np, vton_pcd_np, vton_rgb_np
        # plt.imshow( out_dict['real_rgb_np'])
        show_axes([out_dict['smpl_rast'][0][0].cpu(), out_dict['raster_mask_np'], raster_features_np[..., ::-1],
                real_rgb_nos_np, out_dict['segm_np'], rgb_np,
                real_rgb_nos_np, vton_pcd_np, vton_rgb_np])
        
        # plt.imshow(real_rgb_nos_np)
        # out_img = f'out/appearance/ori_itw_3_real_rgb_nos_np.png'
        # plt.savefig(out_img)


        fig = plt.figure()
        plt.imshow(vton_rgb_np)

        # MODIFY 
        out_img = f'out/appearance/{pose_pid}_output.png'
        plt.savefig(out_img)
        
        print(f"Output image save in : {out_img} \n")
