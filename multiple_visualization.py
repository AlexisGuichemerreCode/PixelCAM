from calendar import EPOCH
from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import subprocess
from pathlib import Path
import datetime as dt
import argparse
# import more_itertools as mit
from dlib.sf_uda import adadsa
import torch.nn as nn

import numpy as np
import numpy
from tqdm import tqdm
# import pretrainedmodels.utils
import yaml
import munch
import pickle
# from texttable import Texttable
from dlib.datasets.wsol_data_core import get_mask
from PIL import Image
import torch
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt

# root_dir = dirname(dirname(dirname(abspath(__file__))))
root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)
import torch.nn.functional as F

from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
from  dlib.configure import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_tag
from dlib.utils.tools import Dict2Obj
# from dlib.utils.tools import log_device
from dlib.configure import config

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model, get_pretrainde_classifier

from dlib.datasets.wsol_loader import get_data_loader
from dlib.datasets.wsol_loader import configure_metadata
from dlib.datasets.wsol_loader import get_class_labels
from dlib.datasets.wsol_loader import get_image_ids
from dlib.learning.train_wsol import Basic, PerformanceMeter
from dlib.utils.tools import get_cpu_device
from dlib.process.parseit import str2bool
from dlib.utils.tools import t2n
import cv2
import json
from glob import glob

def cl_forward(args, model, images):

    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output
    else:
        raise NotImplementedError

    return cl_logits
    
def _compute_accuracy(args, model, loader):
    num_correct = 0
    num_images = 0

    for i, (images, targets, _, _, _, _, _, _) in enumerate(loader):
        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(args, model, images)
            pred = cl_logits.argmax(dim=1)

        num_correct += (pred == targets).sum().item()
        num_images += images.size(0)

    classification_acc = num_correct / float(num_images) * 100
    return classification_acc

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# def numpy_scalar_constructor(loader, node):
#     return float(loader.construct_scalar(node))

# yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', numpy_scalar_constructor)

class IgnoreKeyLoader(yaml.SafeLoader):
    def ignore_keys(self, node):
        ignore_key = 'best_valid_tau_cl'
        if isinstance(node, yaml.MappingNode):
            i = 0
            while i < len(node.value):
                if node.value[i][0].value == ignore_key:
                    del node.value[i]
                else:
                    i += 1
        return self.construct_yaml_map(node)

    def ignore_numpy_scalars(self, node):
        return None  # or any other dummy value

IgnoreKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, IgnoreKeyLoader.ignore_keys)
IgnoreKeyLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar', IgnoreKeyLoader.ignore_numpy_scalars)

def get_visaualization(exp_path, target_method, sf_uda_source_folder, checkpoint_type, dataset, cudaid, image_ids_to_draw, split='test', tmp_outd='tmp_outd', parsedargs=None):
    # config_model.yaml
    # if parsedargs.draw_vis_with_best_source_classifier:
    #     print('wait')
    #     exp_path = exp_path.replace('models_benchmark', 'model_benchmarks_classification')
    # if target_method != 'SOURCE':
    #     tmp_exp_path = glob(f'{exp_path}/*/')
    #     assert len(tmp_exp_path) == 1
    #     exp_path = tmp_exp_path[0]
    if target_method == 'SOURCE' and parsedargs.draw_vis_with_best_source_classifier:
        checkpoint_type = constants.BEST_CL

    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        # args_dict = yaml.safe_load(fy)
        args_dict['model']['freeze_encoder'] = False
        args_dict['pixel_wise_classification'] = False
        #args_dict['spatial_dropout'] = 0.0
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type
        
        # assert dataset == args.dataset, f"dataset name in config file is {args.dataset} but you passed {dataset}"
    
    os.makedirs(args.outd, exist_ok=True)

    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    tag = get_tag(args, checkpoint_type=checkpoint_type)

    msg = 'Task: {} \t box_v2_metric: {} \t' \
        'Dataset: {} \t Method: {} \t ' \
        'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
                                args.method, args.model['encoder_name'])
    encoder_name = args.model['encoder_name']
    method_name = args.method

    
    # DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(cudaid))
    
    tag = get_tag(args, checkpoint_type=checkpoint_type)
    # path_cl = join(exp_path, tag)
    # args.sf_uda_source_folder = path_cl
    # config_model.yaml
    tag = get_tag(args, checkpoint_type=checkpoint_type)
    path_cl = join(exp_path, tag)
    with open(join(path_cl, 'config_model.yaml'), 'r') as fy:
        args_dict = yaml.load(fy, Loader=IgnoreKeyLoader)
        if target_method == 'EnergyCAM':
            args_dict['pixel_wise_classification'] = True
        else:
            args_dict['pixel_wise_classification'] = False
        args_dict['model']['spatial_dropout'] = 0.0
        if target_method == 'NEGEV':
            args_dict['model']['folder_pre_trained_cl'] = "/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/pretrained/GLAS-0-resnet50-GradCAMpp-WGAP-cp_best_classification"
        # args_dict = yaml.safe_load(fy)
        # args_dict['model']['freeze_encoder'] = False
        #args_dict['model']['support_background'] = True
        args = Dict2Obj(args_dict)
        args.outd = tmp_outd
        args.distributed = False
        args.eval_checkpoint_type = checkpoint_type

    args.sf_uda = False
    # args.sf_uda_source_folder = '/export/livia/home/vision/Aguichemerre/models_benchmark/source_models/BLOC/CAMELYON512/id_source_12_CAMELYON512_DEEPMIL_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50/CAMELYON512-0-resnet50-DEEPMIL-DeepMil-cp_best_localization'
    #load source model
    if target_method == 'ADADSA':
        # sf_uda_source_folder = '/export/livia/home/vision/Aguichemerre/models_benchmark/source_models/BLOC/CAMELYON512/id_source_12_CAMELYON512_DEEPMIL_5-tsk_STD_CL-ds_CAMELYON512-fold_0-mag_None-runmode_search-mode-mth_DEEPMIL-spooling_DeepMil-arch_STDClassifier-ecd_resnet50/CAMELYON512-0-resnet50-DEEPMIL-DeepMil-cp_best_localization'
        if dataset == constants.GLAS:
            tag_targ_method = tag.replace(constants.GLAS, constants.CAMELYON512)
        else:
            tag_targ_method = tag.replace(constants.CAMELYON512, constants.GLAS)
        with open(join(sf_uda_source_folder, tag_targ_method, 'config_model.yaml'), 'r') as fy:
            args_dict_source = yaml.load(fy, Loader=IgnoreKeyLoader)
            # args_dict = yaml.safe_load(fy)
            # args_dict['model']['freeze_encoder'] = False
            args_source = Dict2Obj(args_dict_source)
            args_source.outd = tmp_outd
            args_source.distributed = False
            args_source.eval_checkpoint_type = checkpoint_type
        args_source.model['folder_pre_trained_cl'] = os.path.join(sf_uda_source_folder, tag_targ_method)
        model_soruce = get_pretrainde_classifier(args_source)

        s_model = deepcopy(model_soruce)
        t_model = deepcopy(model_soruce)
        t_model = t_model.to(get_cpu_device())
        model = deepcopy(model_soruce)

        adadsa.freeze_all_params(s_model)
        adadsa.freeze_all_params(t_model)
        adadsa.freeze_all_params(model)
        _device = next(model.parameters()).device

        for batch_norm_cl in [nn.BatchNorm1d,
                                nn.BatchNorm2d,
                                nn.BatchNorm3d]:
            adadsa.replace_all_bn_with_adadsa_bn(model=model,
                                                    s_model=s_model,
                                                    t_model=t_model,
                                                    batch_norm_cl=batch_norm_cl,
                                                    device=_device
                                                    )

        model = adadsa.adadsa_freeze_all_model_except_bn_a(model)
    else:
        model = get_model(args)[0]

    print(f'Loading model for {method_name}-{encoder_name} from {path_cl}')
    # if "tscam" in encoder_name:
    #     model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

    #     model.load_state_dict(model_tscam, strict=True)
    # else:
    if target_method == 'EnergyCAM':
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                            map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

        header_p = torch.load(join(path_cl, 'pixel_wise_classification_head.pt'),
                            map_location=get_cpu_device())
        model.pixel_wise_classification_head.load_state_dict(header_p, strict=True)

    elif target_method == 'NEGEV':
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                            map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

        decoder_w = torch.load(join(path_cl, 'decoder.pt'),
                            map_location=get_cpu_device())
        model.decoder.super_load_state_dict(decoder_w, strict=True)

        seg_head_w = torch.load(join(path_cl, 'segmentation_head.pt'),
                            map_location=get_cpu_device())
        model.segmentation_head.load_state_dict(seg_head_w, strict=True)

    else:
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                            map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                            map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)



    DLLogger.log(fmsg("Model checkpoint Loaded from {}".format(path_cl)))
        
    model.to(device)
    model.eval()

    # basic_config = config.get_config(ds=args.dataset, fold=args.fold, magnification=args.magnification)
    basic_config = config.get_config(ds=constants.CAMELYON512, fold=args.fold, magnification=args.magnification)

    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']
    args.cam_curve_interval = basic_config['cam_curve_interval']
    
    ####################################################################################
    ###############load performance log file from orignal exp checkpoint ###############
    ####################################################################################
    assert split == constants.TESTSET or split == constants.VALIDSET
    #split == constants.VALIDSET
    #split = constants.VALIDSET
    log_file_path_best_loc = os.path.join(exp_path, f'performance_log_{checkpoint_type}.pickle')
    if os.path.isfile(log_file_path_best_loc):
        with open(log_file_path_best_loc, 'rb') as f:
            results = pickle.load(f)
        # if split == constants.TESTSET:
        #     clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][-1]
        #     loc_acc_from_orginal_exp_path = results[split]['localization_IOU_50']['value_per_epoch'][-1]
        # else:
        best_epoch = -1 if split == constants.TESTSET else results[split][checkpoint_type.replace('best_', '')]['best_epoch']
        clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][best_epoch]
        loc_acc_from_orginal_exp_path = results[split]['localization']['value_per_epoch'][best_epoch]
        
    ####################################################################################
    ####################################################################################
    DLLogger.flush()
    
    metadata_root = join(constants.RELATIVE_META_ROOT, dataset, f"fold-{args.fold}")
    #read sys var DATASETSH
    args_dict['data_root'] = os.path.join(os.environ['DATASETSH'], 'datasets')
    target_domain_data_paths = config.configure_data_paths(args_dict, dataset)

    loaders = get_data_loader(
            data_roots=target_domain_data_paths,
            metadata_root=metadata_root,
            batch_size=256,#args.batch_size,
            workers=args.num_workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            proxy_training_set=args.proxy_training_set,
            num_val_sample_per_class=args.num_val_sample_per_class,
            std_cams_folder=args.std_cams_folder,
            # distributed_eval=False,
            get_splits_eval=[constants.TESTSET],
            eval_batch_size = 256#args.eval_batch_size,
        )
    
    # t0 = dt.datetime.now()
    # accuracy = _compute_accuracy(args, model, loaders[split])
    # fmsg_tmp = f'Results using split {split} and using best checkpoint that was selected using {checkpoint_type}\n'
    # fmsg_tmp += '\nClassification accuracy from current eval: {}'.format(accuracy)
    # fmsg_tmp += "\nClcassifier's evalaition time of {} split: {}".format(split, dt.datetime.now() - t0)
    # DLLogger.log(fmsg(fmsg_tmp))
    # DLLogger.flush()
    
    cam_computer = CAMComputer(
            args=deepcopy(args),
            model=model,
            loader=loaders[split],
            metadata_root=os.path.join(metadata_root, split),
            mask_root=args.mask_root,
            iou_threshold_list=args.iou_threshold_list,
            dataset_name=args.dataset,
            split=split,
            cam_curve_interval=args.cam_curve_interval,
            multi_contour_eval=args.multi_contour_eval,
            out_folder=args.outd,
        )
    overlay_images = {}
    input_images = {}
    gt_masks = {}
    for batch_idx, (images, targets, _, image_ids, _, _, _, _) in tqdm(
        enumerate(loaders[split]), ncols=constants.NCOLS,
        total=len(loaders[split])):
        image_size = images.shape[2:]
        images = images.to(device)
        targets = targets.to(device)

        for image, target, image_id in zip(images, targets, image_ids):
            if image_id not in image_ids_to_draw:
                continue
            with torch.set_grad_enabled(cam_computer.req_grad):
                cam, cl_logits = cam_computer.get_cam_one_sample(
                    image=image.unsqueeze(0), target=target.item())

            with torch.no_grad():
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)
                cam = cam.detach()
                image_norm = image.permute(1,2,0)
                image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min())
                overlay_image = show_cam_on_image(t2n(image_norm), t2n(cam), use_rgb=True)
                overlay_images[image_id] = overlay_image
                input_images[image_id] = t2n(image_norm)


                # gt_mask = get_mask('f/export/livia/home/vision/Aguichemerre/datasets/{dataset}',
                #            cam_computer.evaluator.mask_paths[image_id],
                #            cam_computer.evaluator.ignore_paths[image_id])
                if target == 1:
                    gt_annotation =  Image.open(os.path.join(f'/export/gauss/vision/Aguichemerre/datasets/{dataset}', cam_computer.evaluator.mask_paths[image_id][0]))
                    gt_annotation = gt_annotation.resize(image_size)
                    gt_annotation=np.asarray(gt_annotation)
                    gt_annotation = (gt_annotation > 0).astype(np.uint8)
                    gt_masks[image_id]=np.asarray(gt_annotation)
                else:
                    gt_masks[image_id] = np.zeros(image_size, dtype=np.uint8)

    return overlay_images, input_images, method_name, gt_masks

def fast_eval():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--pixel_wise_classification", type=str2bool, default=False)
    # parser.add_argument("--exp_path", type=str, default=None)
    parser.add_argument("--tmp_outd", type=str, default='tmp_outd')
    parser.add_argument('--noise_level_for_eval_with_noisy_bbox', nargs='+',
                        type=int, default=[5, 10, 15, 20, 25, 30, 35 ,40, 45, 50])
    parser.add_argument("--target_dataset", type=str, default=None,
                        help="Name of the dataset.", required=True, choices=[constants.CAMELYON512, constants.GLAS])
    parser.add_argument("--path_pre_trained_source", type=str, default=None, help="Path to the pre-trained source model.")
    parser.add_argument('--target_domain_exp_path', type=json.loads, default={})
    parser.add_argument('--image_ids_to_draw', nargs='+', type=str, default=None)
    parser.add_argument('--draw_vis_with_best_source_classifier', type=str2bool, default=False)

    parsedargs = parser.parse_args()
    
    # exp_path = parsedargs.exp_path
    # checkpoint_type = parsedargs.checkpoint_type
    # tmp_outd = join(parsedargs.tmp_outd, os.path.split(exp_path)[-1])#, 'split_'+split+'_'+checkpoint_type)
    # tmp_outd = join(exp_path, '0_re-eval_log')#, 'split_'+split+'_'+checkpoint_type)
    # os.makedirs(tmp_outd, exist_ok=True)
    # # tmp_outd = parsedargs.tmp_outd
    # assert os.path.isdir(exp_path)
    
    os.makedirs('tmp_outd', exist_ok=True)  
    log_backends = [
                # ArbJSONStreamBackend(Verbosity.VERBOSE,
                #                     join(args.outd, "log.json")),
                ArbTextStreamBackend(Verbosity.VERBOSE,
                                    join('tmp_outd', f"split_{parsedargs.split}_log.txt")),
            ]
    
    _VERBOSE = True
    if _VERBOSE:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
        
    DLLogger.GLOBAL_LOGGER = DLLogger.NotInitializedObject()
        
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())
    ##########

    base_checkpoint_types = [constants.BEST_LOC]
        
    for checkpoint_type_extended in base_checkpoint_types:
        checkpoint_type = checkpoint_type_extended
        
        split = parsedargs.split
        # exp_path = parsedargs.exp_path
        # # checkpoint_type = parsedargs.checkpoint_type
        # # tmp_outd = join(parsedargs.tmp_outd, os.path.split(exp_path)[-1])#, 'split_'+split+'_'+checkpoint_type)
        # tmp_outd = join(exp_path, '0_re-eval_log')#, 'split_'+split+'_'+checkpoint_type)
        # # tmp_outd = parsedargs.tmp_outd
        # assert os.path.isdir(exp_path)
        assert split == constants.TESTSET or split == constants.VALIDSET, split
        
        _CODE_FUNCTION = 'fast_eval_{}'.format(split)

        target_methods = ['DeepMIL','EnergyCAM']
        #'CAM', 'GradCAMpp', 'NEGEV',
        # target_methods = ['ADADSA']GradCAMpp'EnergyCAM', 'NEGEV', 
        #create fig len(parsedargs.image_ids_to_draw) row and len(target_methods) columns
        fig, axs = plt.subplots(len(parsedargs.image_ids_to_draw), len(target_methods)+2, figsize=((len(target_methods)+2)*1.9, 2*len(parsedargs.image_ids_to_draw)))

        method_name_lst = []
        for ind_method, target_method in enumerate(target_methods):
            ind_method+= 2
            # if target_method == 'ADADSA':
            #     for i, image_id in enumerate(parsedargs.image_ids_to_draw):
            #         axs[i, ind_method].axis('off')
            #         if ind_method == 1:
            #             axs[i, ind_method].set_title(f'{target_method}')
            #     continue

            exp_path = parsedargs.target_domain_exp_path[target_method]
            overlay_images, input_images, method_name, gt_masks = get_visaualization(exp_path=exp_path, target_method=target_method, sf_uda_source_folder=parsedargs.path_pre_trained_source, checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split='test', tmp_outd='tmp_outd', parsedargs=parsedargs)
            method_name_lst.append(method_name)
            #add images to fig
            for i, image_id in enumerate(parsedargs.image_ids_to_draw):
                overlay_image = overlay_images[image_id]
                
                axs[i, ind_method].imshow(overlay_image)
                if i == 0:
                    methd_name_to_print = method_name.upper().replace('GRADCAMPP', 'GRADCAM++')
                    axs[i, ind_method].set_title(f'{target_method}', fontsize=18)
                axs[i, ind_method].axis('off')
                #dhow input image in the first column
                input_image = input_images[image_id]
                axs[i, 0].imshow(input_image)
                axs[i, 0].axis('off')

                axs[i, 1].imshow(gt_masks[image_id], cmap='gray')
                axs[i, 1].axis('off')
                if i == 0:
                    axs[i, 0].set_title('INPUT\nTARGET', fontsize=18)
                
                if i == 0:
                    axs[i, 1].set_title('MASK', fontsize=18)
        #check all values are same in encoder_name_lst
        # assert len(set(method_name_lst)) == 1
                    
        plt.tight_layout()
        # fig.subplots_adjust(wspace=0.1)
        out_dir = 'visualization'
        os.makedirs(out_dir, exist_ok=True)

        out_dir = os.path.join(out_dir, f'{checkpoint_type}')
        os.makedirs(out_dir, exist_ok=True)
        
        out_dir = os.path.join(out_dir, f'source_domain_{parsedargs.target_dataset}')
        os.makedirs(out_dir, exist_ok=True)

        #out_dir = os.path.join(out_dir, f'normal')
        #os.makedirs(out_dir, exist_ok=True)

        #plt.savefig(os.path.join(out_dir, f'vis_{method_name_lst[0]}_{checkpoint_type}_target_ds_{parsedargs.target_dataset}_with_source_best_cl_{parsedargs.draw_vis_with_best_source_classifier}.png'))
        plt.savefig(os.path.join(out_dir, 'normal.png'))
        # overlay_images = get_visaualization(exp_path=parsedargs.path_pre_trained_source, checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split='test', tmp_outd='tmp_outd')
        # overlay_images = get_visaualization(exp_path=parsedargs.target_domain_exp_path['SFDE'], checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split='test', tmp_outd='tmp_outd')
        # overlay_images = get_visaualization(exp_path=parsedargs.target_domain_exp_path['SHOT'], checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split='test', tmp_outd='tmp_outd')
        # overlay_images = get_visaualization(exp_path=parsedargs.target_domain_exp_path['CDCL'], checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split='test', tmp_outd='tmp_outd')
        # overlay_images = get_visaualization(exp_path=parsedargs.target_domain_exp_path['ADADSA'], checkpoint_type=checkpoint_type, dataset=parsedargs.target_dataset, cudaid=parsedargs.cudaid, image_ids_to_draw=parsedargs.image_ids_to_draw, split='test', tmp_outd='tmp_outd')
        ########################################
        ########################################
        ########################################

        
    #     with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
    #         args_dict = yaml.safe_load(fy)
    #         args_dict['model']['freeze_encoder'] = False
    #         args = Dict2Obj(args_dict)
    #         args.outd = tmp_outd
    #         args.distributed = False
    #         args.eval_checkpoint_type = checkpoint_type
            
    #         assert parsedargs.dataset == args.dataset, f"dataset name in config file is {args.dataset} but you passed {parsedargs.dataset}"
        
    #     os.makedirs(args.outd, exist_ok=True)

    #     _DEFAULT_SEED = args.MYSEED
    #     os.environ['MYSEED'] = str(args.MYSEED)

    #     tag = get_tag(args, checkpoint_type=checkpoint_type)

    #     msg = 'Task: {} \t box_v2_metric: {} \t' \
    #         'Dataset: {} \t Method: {} \t ' \
    #         'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
    #                                 args.method, args.model['encoder_name'])

        
    #     DLLogger.log(fmsg("Start time: {}".format(t0)))
    #     DLLogger.log(fmsg(msg))

    #     set_seed(seed=_DEFAULT_SEED, verbose=False)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    #     device = torch.device('cuda:{}'.format(parsedargs.cudaid))
        
    #     model = get_model(args)[0]
    #     tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
    #     path_cl = join(exp_path, tag)

    #     if "tscam" in parsedargs.encoder_name:
    #         model_tscam = torch.load(join(path_cl, 'model.pt'),map_location=get_cpu_device())

    #         model.load_state_dict(model_tscam, strict=True)
    #     else:
    #         encoder_w = torch.load(join(path_cl, 'encoder.pt'),
    #                             map_location=get_cpu_device())
    #         model.encoder.super_load_state_dict(encoder_w, strict=True)

    #         header_w = torch.load(join(path_cl, 'classification_head.pt'),
    #                             map_location=get_cpu_device())
    #         model.classification_head.load_state_dict(header_w, strict=True)

    #     DLLogger.log(fmsg("Model checkpoint Loaded from {}".format(path_cl)))
            
    #     model.to(device)
    #     model.eval()

    #     # basic_config = config.get_config(ds=args.dataset, fold=args.fold, magnification=args.magnification)
    #     basic_config = config.get_config(ds=constants.CAMELYON512, fold=args.fold, magnification=args.magnification)

    #     args.data_paths = basic_config['data_paths']
    #     args.metadata_root = basic_config['metadata_root']
    #     args.mask_root = basic_config['mask_root']
    #     args.cam_curve_interval = basic_config['cam_curve_interval']
        
    #     ####################################################################################
    #     ###############load performance log file from orignal exp checkpoint ###############
    #     ####################################################################################
    #     assert split == constants.TESTSET or split == constants.VALIDSET
    #     #split == constants.VALIDSET
    #     #split = constants.VALIDSET
    #     log_file_path_best_loc = os.path.join(exp_path, f'performance_log_{checkpoint_type}.pickle')
    #     if os.path.isfile(log_file_path_best_loc):
    #         with open(log_file_path_best_loc, 'rb') as f:
    #             results = pickle.load(f)
    #         # if split == constants.TESTSET:
    #         #     clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][-1]
    #         #     loc_acc_from_orginal_exp_path = results[split]['localization_IOU_50']['value_per_epoch'][-1]
    #         # else:
    #         best_epoch = -1 if split == constants.TESTSET else results[split][checkpoint_type.replace('best_', '')]['best_epoch']
    #         clas_acc_from_orginal_exp_path = results[split]['classification']['value_per_epoch'][best_epoch]
    #         loc_acc_from_orginal_exp_path = results[split]['localization']['value_per_epoch'][best_epoch]
            
    #     ####################################################################################
    #     ####################################################################################
    #     DLLogger.flush()
        
    #     loaders = get_data_loader(
    #             data_roots=args.data_paths,
    #             metadata_root=args.metadata_root,
    #             batch_size=args.batch_size,
    #             workers=args.num_workers,
    #             resize_size=args.resize_size,
    #             crop_size=args.crop_size,
    #             proxy_training_set=args.proxy_training_set,
    #             num_val_sample_per_class=args.num_val_sample_per_class,
    #             std_cams_folder=args.std_cams_folder,
    #             # distributed_eval=False,
    #             get_splits_eval=[constants.TESTSET],
    #             eval_batch_size = args.eval_batch_size,
    #         )
        
    #     t0 = dt.datetime.now()
    #     accuracy = _compute_accuracy(args, model, loaders[split])
    #     fmsg_tmp = f'Results using split {split} and using best checkpoint that was selected using {checkpoint_type}\n'
    #     fmsg_tmp += '\nClassification accuracy from current eval: {}'.format(accuracy)
    #     fmsg_tmp += "\nClcassifier's evalaition time of {} split: {}".format(split, dt.datetime.now() - t0)
    #     DLLogger.log(fmsg(fmsg_tmp))
    #     DLLogger.flush()
        
    #     cam_computer = CAMComputer(
    #             args=deepcopy(args),
    #             model=model,
    #             loader=loaders[split],
    #             metadata_root=os.path.join(args.metadata_root, split),
    #             mask_root=args.mask_root,
    #             iou_threshold_list=args.iou_threshold_list,
    #             dataset_name=args.dataset,
    #             split=split,
    #             cam_curve_interval=args.cam_curve_interval,
    #             multi_contour_eval=args.multi_contour_eval,
    #             out_folder=args.outd,
    #         )
        
    #     for batch_idx, (images, targets, _, image_ids, _, _, _, _) in tqdm(
    #         enumerate(loaders[split]), ncols=constants.NCOLS,
    #         total=len(loaders[split])):
    #         image_size = images.shape[2:]
    #         images = images.to(device)
    #         targets = targets.to(device)

    #         for image, target, image_id in zip(images, targets, image_ids):
    #             with torch.set_grad_enabled(cam_computer.req_grad):
    #                 cam, cl_logits = cam_computer.get_cam_one_sample(
    #                     image=image.unsqueeze(0), target=target.item())

    #             with torch.no_grad():
    #                 cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
    #                                     image_size,
    #                                     mode='bilinear',
    #                                     align_corners=False).squeeze(0).squeeze(0)
    #                 cam = cam.detach()
    #                 image_norm = image.permute(1,2,0)
    #                 image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min())
    #                 overlay_image = show_cam_on_image(t2n(image_norm), t2n(cam))

    #         cam = t2n(cam)

    #     t0 = dt.datetime.now()
    #     cam_performance = cam_computer.compute_and_evaluate_cams()
    #     print(cam_computer.evaluator.perf_gist)
    #     loc_score = cam_performance
    #     fmsg_tmp = f'Results using split {split} and using best checkpoint that was selected using {checkpoint_type}\n'
    #     fmsg_tmp +=  '\nLocalization accuracy from current eval: {}'.format(loc_score)
        
    #     DLLogger.log(fmsg(fmsg_tmp))
    #     DLLogger.log(fmsg('BYE BYE'))
    #     DLLogger.flush()
    # print('wait')
    
    
if __name__ == '__main__':
    fast_eval()