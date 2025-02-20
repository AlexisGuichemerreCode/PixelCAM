import copy
import warnings
import sys
import os
from os.path import dirname, abspath, join, basename
from copy import deepcopy
from typing import Iterable, Union

import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.learning import lr_scheduler as my_lr_scheduler

from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import count_nb_params
from dlib.configure import constants
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag
from dlib.utils.shared import format_dict_2_str

from dlib.sf_uda import adadsa

from dlib.generation import cgan
from dlib.models_parts import uda_backprop

import dlib
from dlib import create_model

from dlib.losses.elb import ELB
from dlib import losses
from dlib.losses import sf_uda_sdda


import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = [
    'get_loss',
    'get_pretrainde_classifier',
    'get_model',
    'get_optimizer_of_model',
    'get_optimizer_for_params'
]


def freeze_all_params(model):
    for module in (model.modules()):

        for param in module.parameters():
            param.requires_grad = False

        if isinstance(module, torch.nn.BatchNorm3d):
            module.eval()

        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()

        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()

        if isinstance(module, torch.nn.Dropout):
            module.eval()


def get_negev_loss(args, masterloss):
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag

    assert args.dataset in [constants.CAMELYON512, constants.GLAS]

    if not args.model['freeze_cl']:
        masterloss.add(losses.ClLoss(
            cuda_id=args.c_cudaid,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
              mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

    if args.crf_ng:
        masterloss.add(losses.ConRanFieldNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.crf_ng_lambda,
            sigma_rgb=args.crf_ng_sigma_rgb,
            sigma_xy=args.crf_ng_sigma_xy,
            scale_factor=args.crf_ng_scale,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.crf_ng_start_ep,
            end_epoch=args.crf_ng_end_ep,
        ))

    if args.jcrf_ng:
        ljcrf = losses.JointConRanFieldNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.jcrf_ng_lambda,
            sigma_rgb=args.jcrf_ng_sigma_rgb,
            scale_factor=args.jcrf_ng_scale,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.jcrf_ng_start_ep,
            end_epoch=args.jcrf_ng_end_ep
        )
        ljcrf.set_it(pair_mode=args.jcrf_ng_pair_mode, n=args.jcrf_ng_n,
                     dataset_name=args.dataset)
        masterloss.add(ljcrf)

    if args.max_sizepos_ng:
        size_loss = losses.MaxSizePositiveNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.max_sizepos_ng_lambda,
            elb=deepcopy(elb),
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.max_sizepos_ng_start_ep,
            end_epoch=args.max_sizepos_ng_end_ep
        )
        size_loss.set_it(apply_negative_samples=not constants.DS_HAS_NEG_SAM[
            args.dataset], negative_c=constants.DS_NEG_CL[args.dataset])
        masterloss.add(size_loss)

    if args.neg_samples_ng:
        lnegs = losses.NegativeSamplesNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.neg_samples_ng_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.neg_samples_ng_start_ep,
            end_epoch=args.neg_samples_ng_end_ep
        )
        lnegs.set_it(negative_c=constants.DS_NEG_CL[args.dataset])
        masterloss.add(lnegs)

    if args.sl_ng:
        sl_loss = losses.SelfLearningNegev(
            cuda_id=args.c_cudaid,
            lambda_=args.sl_ng_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.sl_ng_start_ep,
            end_epoch=args.sl_ng_end_ep,
            seg_ignore_idx=args.seg_ignore_idx
        )
        sl_loss.set_it(apply_negative_samples=not constants.DS_HAS_NEG_SAM[
            args.dataset], negative_c=constants.DS_NEG_CL[args.dataset])

        masterloss.add(sl_loss)

    return masterloss


def get_encoder_d_c(encoder_name):
    if encoder_name in [constants.VGG16]:
        vgg_encoders = dlib.encoders.vgg_encoders
        encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        decoder_channels = (256, 128, 64)
    else:
        encoder_depth = 5
        decoder_channels = (256, 128, 64, 32, 16)

    return encoder_depth, decoder_channels


def get_loss(args):

    if args.sf_uda:
        return get_loss_target(args)

    else:
        return get_loss_source(args)


def sfuda_sdda_get_adv_discriminator_loss(args):
    """
    Returns the adversarial loss of the discriminator.
    Mandatory
    :param args:
    :return:
    """
    assert args.sf_uda
    assert args.sdda
    assert args.adv_d_sdda  # mandatory

    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    if args.task == constants.STD_CL:
        adv_d_loss = sf_uda_sdda.UdaSddaAdvDiscriminator(
            cuda_id=args.c_cudaid,
            lambda_=args.adv_d_sdda_lambda,
            support_background=support_background,
            multi_label_flag=multi_label_flag,
            start_epoch=args.adv_d_sdda_start_ep,
            end_epoch=args.adv_d_sdda_end_ep
        )
        masterloss.add(adv_d_loss)

    elif args.task == constants.NEGEV:
        raise NotImplementedError  # todo

    else:
        raise NotImplementedError

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    DLLogger.log(message=f"Train loss (SDDA) - ADV. DISCRIMINATOR:"
                         f" {masterloss}")

    return masterloss


def sfuda_sdda_get_generation_adaptation_loss(args):
    """
    Generation loss: Adv generator, source model prediction CE, max p(x) of
    source model.
    w.r.t Generator params.

    Adaptation loss: trg classifier CE, Domain discriminator.
    w.r.t generator + trg_model + domain_discriminator.
    :param args:
    :return:
    """
    assert args.sf_uda
    assert args.sdda
    assert args.adv_g_sdda  # mandatory.

    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    if args.task == constants.STD_CL:
        # Generation --------
        if args.adv_g_sdda:
            adv_g_loss = sf_uda_sdda.UdaSddaAdvGenerator(
                cuda_id=args.c_cudaid,
                lambda_=args.adv_g_sdda_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.adv_g_sdda_start_ep,
                end_epoch=args.adv_g_sdda_end_ep
            )
            masterloss.add(adv_g_loss)

        if args.px_sdda:
            px_loss = sf_uda_sdda.UdaSddaSrcModelPxLikelihood(
                cuda_id=args.c_cudaid,
                lambda_=args.px_sdda_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.px_sdda_start_ep,
                end_epoch=args.px_sdda_end_ep
            )
            masterloss.add(px_loss)

        if args.ce_src_m_fake_sdda:
            ce_src_ce_fake_loss = sf_uda_sdda.UdaSddaSrcModelCeFakeImage(
                cuda_id=args.c_cudaid,
                lambda_=args.ce_src_m_fake_sdda_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.ce_src_m_fake_sdda_start_ep,
                end_epoch=args.ce_src_m_fake_sdda_end_ep
            )
            masterloss.add(ce_src_ce_fake_loss)

        # Adaptation --------
        if args.ce_trg_m_fake_sdda:
            if args.method == constants.METHOD_SPG:
                ce_trg_ce_fake_loss = sf_uda_sdda.SpgUdaSddaTrgModelCeFakeImage(
                    cuda_id=args.c_cudaid,
                    lambda_=args.ce_trg_m_fake_sdda_lambda,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag,
                    start_epoch=args.adaptation_start_epoch,
                    end_epoch=args.ce_trg_m_fake_sdda_end_ep
                )
                ce_trg_ce_fake_loss.spg_threshold_1h = args.spg_threshold_1h
                ce_trg_ce_fake_loss.spg_threshold_1l = args.spg_threshold_1l
                ce_trg_ce_fake_loss.spg_threshold_2h = args.spg_threshold_2h
                ce_trg_ce_fake_loss.spg_threshold_2l = args.spg_threshold_2l
                ce_trg_ce_fake_loss.spg_threshold_3h = args.spg_threshold_3h
                ce_trg_ce_fake_loss.spg_threshold_3l = args.spg_threshold_3l
                ce_trg_ce_fake_loss.hyper_p_set = True
                ce_trg_ce_fake_loss.set_it(ce_label_smoothing=0.0)

            elif args.method == constants.METHOD_ACOL:
                ce_trg_ce_fake_loss = \
                    sf_uda_sdda.AcolUdaSddaTrgModelCeFakeImage(
                        cuda_id=args.c_cudaid,
                        lambda_=args.ce_trg_m_fake_sdda_lambda,
                        support_background=support_background,
                        multi_label_flag=multi_label_flag,
                        start_epoch=args.adaptation_start_epoch,
                        end_epoch=args.ce_trg_m_fake_sdda_end_ep
                    )
                ce_trg_ce_fake_loss.set_it(ce_label_smoothing=0.0)

            elif args.method == constants.METHOD_CUTMIX:
                ce_trg_ce_fake_loss = \
                    sf_uda_sdda.CutMixUdaSddaTrgModelCeFakeImage(
                        cuda_id=args.c_cudaid,
                        lambda_=args.ce_trg_m_fake_sdda_lambda,
                        support_background=support_background,
                        multi_label_flag=multi_label_flag,
                        start_epoch=args.adaptation_start_epoch,
                        end_epoch=args.ce_trg_m_fake_sdda_end_ep
                    )
                ce_trg_ce_fake_loss.set_it(ce_label_smoothing=0.0)

            elif args.method == constants.METHOD_MAXMIN:
                raise NotImplementedError

            else:
                ce_trg_ce_fake_loss = sf_uda_sdda.UdaSddaTrgModelCeFakeImage(
                    cuda_id=args.c_cudaid,
                    lambda_=args.ce_trg_m_fake_sdda_lambda,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag,
                    start_epoch=args.adaptation_start_epoch,
                    end_epoch=args.ce_trg_m_fake_sdda_end_ep
                )
            masterloss.add(ce_trg_ce_fake_loss)

        if args.ce_dom_d_sdda:
            ce_dom_d_loss = sf_uda_sdda.UdaSddaDomainDiscriminator(
                cuda_id=args.c_cudaid,
                lambda_=1.,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.adaptation_start_epoch,
                end_epoch=args.ce_dom_d_sdda_end_ep
            )
            masterloss.add(ce_dom_d_loss)

    elif args.task == constants.NEGEV:
        raise NotImplementedError  # todo

    else:
        raise NotImplementedError

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    DLLogger.log(message=f"Train loss (SDDA) - Generation + Adaptation:"
                         f" {masterloss}")

    return masterloss


def get_loss_target(args):

    assert args.sf_uda

    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    if args.task == constants.STD_CL:
        if args.ce_pseudo_lb:

            if args.method == constants.METHOD_SPG:
                cl_loss = losses.UdaSpgLoss(
                    cuda_id=args.c_cudaid,
                    lambda_=args.ce_pseudo_lb_lambda,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag
                )
                cl_loss.spg_threshold_1h = args.spg_threshold_1h
                cl_loss.spg_threshold_1l = args.spg_threshold_1l
                cl_loss.spg_threshold_2h = args.spg_threshold_2h
                cl_loss.spg_threshold_2l = args.spg_threshold_2l
                cl_loss.spg_threshold_3h = args.spg_threshold_3h
                cl_loss.spg_threshold_3l = args.spg_threshold_3l
                cl_loss.hyper_p_set = True
                cl_loss.set_it(ce_label_smoothing=args.ce_pseudo_lb_smooth)
                masterloss.add(cl_loss)

            elif args.method == constants.METHOD_ACOL:
                cl_loss = losses.UdaAcolLoss(
                    cuda_id=args.c_cudaid,
                    lambda_=args.ce_pseudo_lb_lambda,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag
                )
                cl_loss.set_it(ce_label_smoothing=args.ce_pseudo_lb_smooth)
                masterloss.add(cl_loss)

            elif args.method == constants.METHOD_CUTMIX:
                cl_loss = losses.UdaCutMixLoss(
                    cuda_id=args.c_cudaid,
                    lambda_=args.ce_pseudo_lb_lambda,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag
                )
                cl_loss.set_it(ce_label_smoothing=args.ce_pseudo_lb_smooth)
                masterloss.add(cl_loss)


            elif args.method == constants.METHOD_MAXMIN:
                raise NotImplementedError

            else:
                ce_loss = losses.UdaCrossEntropyImgPseudoLabels(
                    cuda_id=args.c_cudaid,
                    lambda_=args.ce_pseudo_lb_lambda,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag,
                    start_epoch=args.ce_pseudo_lb_start_ep,
                    end_epoch=args.ce_pseudo_lb_end_ep
                )
                ce_loss.set_it(ce_label_smoothing=args.ce_pseudo_lb_smooth)
                masterloss.add(ce_loss)

        if args.ent_pseudo_lb:
            ent_loss = losses.UdaTargetClassProbEntropy(
                cuda_id=args.c_cudaid,
                lambda_=args.ent_pseudo_lb_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.ent_pseudo_lb_start_ep,
                end_epoch=args.ent_pseudo_lb_end_ep
            )
            masterloss.add(ent_loss)

        if args.div_pseudo_lb:
            div_loss = losses.UdaDiversityTargetClass(
                cuda_id=args.c_cudaid,
                lambda_=args.div_pseudo_lb_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.div_pseudo_lb_start_ep,
                end_epoch=args.div_pseudo_lb_end_ep
            )
            masterloss.add(div_loss)

        # FAUST
        if args.views_ft_consist:
            assert args.faust
            assert args.faust_n_views > 0, args.faust_n_views

            ft_views_loss = losses.UdaFeatureViewsConsistencyFaust(
                cuda_id=args.c_cudaid,
                lambda_=args.views_ft_consist_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.views_ft_consist_start_ep,
                end_epoch=args.views_ft_consist_end_ep
            )
            masterloss.add(ft_views_loss)

        if args.ce_views_soft_pl:
            assert args.faust
            assert args.faust_n_views > 0, args.faust_n_views
            assert args.ce_views_soft_pl_t > 0., args.ce_views_soft_pl_t

            soft_lb_views_loss = losses.UdaClassProbsViewsSoftLabelsFaust(
                cuda_id=args.c_cudaid,
                lambda_=args.ce_views_soft_pl_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.ce_views_soft_pl_start_ep,
                end_epoch=args.ce_views_soft_pl_end_ep
            )
            masterloss.add(soft_lb_views_loss)

        if args.mc_var_prob:
            assert args.faust
            assert args.mc_var_prob_n_dout >= 2, args.mc_var_prob_n_dout

            mc_loss = losses.UdaMcDropoutVarMinFaust(
                cuda_id=args.c_cudaid,
                lambda_=args.mc_var_prob_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.mc_var_prob_start_ep,
                end_epoch=args.mc_var_prob_end_ep
            )
            masterloss.add(mc_loss)

        if args.min_prob_entropy:
            assert args.faust

            entropy_loss = losses.UdaClassProbsEntropyFaust(
                cuda_id=args.c_cudaid,
                lambda_=args.min_prob_entropy_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.min_prob_entropy_start_ep,
                end_epoch=args.min_prob_entropy_end_ep
            )
            masterloss.add(entropy_loss)

        #NRC
        if args.nrc_na:
            na_loss = losses.UdaNANrc(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
            )
            na_loss.set_it(nrc_na_lambda=args.nrc_na_lambda)
            masterloss.add(na_loss)

        if args.nrc_ena:
            ena_loss = losses.UdaENANrc(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
            )
            ena_loss.set_it(nrc_ena_lambda=args.nrc_ena_lambda)
            masterloss.add(ena_loss)
        
        if args.nrc_kl:
            kl_loss = losses.UdaKLNrc(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
            )
            kl_loss.set_it(nrc_kl_lambda=args.nrc_kl_lambda,epsilon=args.nrc_epsilon)
            masterloss.add(kl_loss)

        if args.cdcl_pseudo_lb:
            # model,_  = get_model(args)
            # if support_background:
            #     weights = model.classification_head.fc.weight[1:]
            # else:
            #     weights = model.classification_head.fc.weight

            cdcl_loss = losses.UdaCdcl(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
            )
            cdcl_loss.set_it(tau = args.cdcl_tau, cdcl_lambda = args.cdcl_lambda,)
            masterloss.add(cdcl_loss)

        if args.cdd_pseudo_lb:
            cdd_loss = losses.UdaCdd(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
            )
            cdd_loss.set_it(num_layers=args.cdd_pseudo_lb_num_layers, kernel_num=args.cdd_pseudo_lb_kernel_num,
                            kernel_mul=args.cdd_pseudo_lb_kernel_mul, num_classes=args.num_classes, lambda_=args.cdd_lambda)
            
            masterloss.add(cdd_loss)
            
            
                


    elif args.task == constants.NEGEV:
        raise NotImplementedError  # todo

    else:
        raise NotImplementedError

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    DLLogger.log(message=f"Train loss (SFUDA): {masterloss}")

    return masterloss


def get_loss_source(args):

    assert not args.sf_uda

    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    # image classification loss
    if args.task == constants.STD_CL:
        if args.method == constants.METHOD_SPG:
            cl_loss = losses.SpgLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag)
            cl_loss.spg_threshold_1h = args.spg_threshold_1h
            cl_loss.spg_threshold_1l = args.spg_threshold_1l
            cl_loss.spg_threshold_2h = args.spg_threshold_2h
            cl_loss.spg_threshold_2l = args.spg_threshold_2l
            cl_loss.spg_threshold_3h = args.spg_threshold_3h
            cl_loss.spg_threshold_3l = args.spg_threshold_3l
            cl_loss.hyper_p_set = True
            cl_loss.set_it(ce_label_smoothing=args.ce_label_smoothing)
            masterloss.add(cl_loss)

        elif args.method == constants.METHOD_ACOL:
            cl_loss = losses.AcolLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag)
            cl_loss.set_it(ce_label_smoothing=args.ce_label_smoothing)
            masterloss.add(cl_loss)

        elif args.method == constants.METHOD_SAT:
            masterloss.add(losses.SatLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        elif args.method == constants.METHOD_CUTMIX:
            cl_loss = losses.CutMixLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag)
            cl_loss.set_it(ce_label_smoothing=args.ce_label_smoothing)
            masterloss.add(cl_loss)

        elif args.method == constants.METHOD_MAXMIN:
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)
            loss = losses.MaxMinLoss(
                    cuda_id=args.c_cudaid,
                    elb=elb,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag)
            loss.set_dataset_name(dataset_name=args.dataset)
            loss.set_lambda_neg(lambda_neg=args.minmax_lambda_neg)
            loss.set_lambda_size(lambda_size=args.minmax_lambda_size)
            loss.set_ce_label_smoothing(ce_label_smoothing=args.ce_label_smoothing)
            masterloss.add(loss)

        else:
            if args.model['image_classifier']:
                cl_loss = losses.ClLoss(
                    cuda_id=args.c_cudaid,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag)
                cl_loss.set_it(ce_label_smoothing=args.ce_label_smoothing)
                masterloss.add(cl_loss)

    elif args.task == constants.SEG:
        masterloss.add(losses.SegLoss(
            cuda_id=args.c_cudaid,
            support_background=support_background,
            multi_label_flag=multi_label_flag))

    elif args.task == constants.NEGEV:
        masterloss = get_negev_loss(args, masterloss)
    # fcams
    elif args.task == constants.F_CL:

        if not args.model['freeze_cl']:
            cl_loss = losses.ClLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag)
            cl_loss.set_it(ce_label_smoothing=args.ce_label_smoothing)
            masterloss.add(cl_loss)

        elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                  mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

        if args.im_rec:
            masterloss.add(
                losses.ImgReconstruction(
                    cuda_id=args.c_cudaid,
                    lambda_=args.im_rec_lambda,
                    elb=deepcopy(elb) if args.sr_elb else nn.Identity(),
                    support_background=support_background,
                    multi_label_flag=multi_label_flag)
            )

        if args.crf_fc:
            masterloss.add(losses.ConRanFieldFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.crf_lambda,
                sigma_rgb=args.crf_sigma_rgb, sigma_xy=args.crf_sigma_xy,
                scale_factor=args.crf_scale,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.crf_start_ep, end_epoch=args.crf_end_ep,
            ))

        if args.entropy_fc:
            masterloss.add(losses.EntropyFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.entropy_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        if args.max_sizepos_fc:
            masterloss.add(losses.MaxSizePositiveFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.max_sizepos_fc_lambda,
                elb=deepcopy(elb), support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.max_sizepos_fc_start_ep,
                end_epoch=args.max_sizepos_fc_end_ep
            ))

        if args.sl_fc:
            sl_fcam = losses.SelfLearningFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.sl_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.sl_start_ep, end_epoch=args.sl_end_ep,
                seg_ignore_idx=args.seg_ignore_idx
            )

            masterloss.add(sl_fcam)

        assert len(masterloss.n_holder) > 1
    else:
        raise NotImplementedError
    
    #Pixel classification
    if args.task == constants.STD_CL and args.pixel_wise_classification:
        
        if args.ece:
            EnergyCE_loss = losses.EnergyCEloss(
                    cuda_id=args.c_cudaid,
                    support_background=support_background,
                    multi_label_flag=multi_label_flag,
                    dataset=args.dataset)
            
            if args.dataset == constants.GLAS:
                negative_samples = False
            elif args.dataset == constants.CAMELYON512 and args.neg_samples_partial:
                negative_samples = False
            elif args.dataset == constants.CAMELYON512:
                negative_samples = True
            
            EnergyCE_loss.set_it(ece_lambda=args.ece_lambda, apply_negative_samples=negative_samples, negative_c=constants.DS_NEG_CL[args.dataset])
            masterloss.add(EnergyCE_loss)

        if args.pxortho:
            lpxorth = losses.PxOrtognalityloss(
                cuda_id=args.c_cudaid,
                lambda_=args.neg_samples_ng_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.neg_samples_ng_start_ep,
                end_epoch=args.neg_samples_ng_end_ep
            )
            lpxorth.set_it(pxortho_lambda=args.pxortho_lambda)
            masterloss.add(lpxorth)

        if args.neg_samples_ng:
            lnegs = losses.NegativeSamplesNegev(
                cuda_id=args.c_cudaid,
                lambda_=args.neg_samples_ng_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.neg_samples_ng_start_ep,
                end_epoch=args.neg_samples_ng_end_ep
            )
            lnegs.set_it(negative_c=constants.DS_NEG_CL[args.dataset])
            masterloss.add(lnegs)

        if args.crf_fc:
            masterloss.add(losses.ConRanFieldPxcams(
                cuda_id=args.c_cudaid,
                lambda_=args.crf_lambda,
                sigma_rgb=args.crf_sigma_rgb, sigma_xy=args.crf_sigma_xy,
                scale_factor=args.crf_scale,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.crf_start_ep, end_epoch=args.crf_end_ep,
            ))

        if args.entropy_fc:
            masterloss.add(losses.EntropyFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.entropy_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag))
            
        if args.max_sizepos_fc:
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                  mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)
            
            masterloss.add(losses.MaxSizePositiveFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.max_sizepos_fc_lambda,
                elb=deepcopy(elb), support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.max_sizepos_fc_start_ep,
                end_epoch=args.max_sizepos_fc_end_ep
            ))


        # if args.eng_marginal:
        #     EnergyMarginal_loss = losses.EnergyMGloss(
        #             cuda_id=args.c_cudaid,
        #             support_background=support_background,
        #             multi_label_flag=multi_label_flag)
        #     EnergyMarginal_loss.set_it(eng_lambda=args.eng_lambda)
        #     masterloss.add(EnergyMarginal_loss)

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    DLLogger.log(message="Train loss: {}".format(masterloss))
    return masterloss


def get_aux_params(args):
    """
    Prepare the head params.
    :param args:
    :return:
    """
    assert args.spatial_pooling in constants.SPATIAL_POOLINGS
    return {
        "pooling_head": args.spatial_pooling,
        "classes": args.num_classes,
        "modalities": args.wc_modalities,
        "kmax": args.wc_kmax,
        "kmin": args.wc_kmin,
        "alpha": args.wc_alpha,
        "dropout": args.wc_dropout,
        "support_background": args.model['support_background'],
        "freeze_cl": args.model['freeze_cl'],
        "r": args.lse_r,
        "mid_channels": args.mil_mid_channels,
        "gated": args.mil_gated,
        'prm_ks': args.prm_ks if hasattr(args, 'prm_ks') else 3,
        'prm_st': args.prm_st if hasattr(args, 'prm_st') else 1,
        'pixel_wise_classification' : args.pixel_wise_classification,
        'batch_norm' : args.batch_norm_pixel_classifier,
        'multiple_layer' :  args.multiple_layer_pixel_classifier,
        'one_layer' :  args.one_layer_pixel_classifier,
        'anchors_ortogonal' : args.anchors_ortogonal,
        'detach_pixel_classifier' : args.detach_pixel_classifier
    }


def get_pretrainde_classifier(args):
    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL, constants.METHOD_TSCAM,
                constants.METHOD_SAT]

    if args.method in spec_mth:
        if args.method == constants.METHOD_ACOL:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                num_classes=args.num_classes,
                acol_drop_threshold=args.acol_drop_threshold,
                large_feature_map=args.acol_large_feature_map,
                scale_in=p.scale_in
            )
        elif args.method == constants.METHOD_SPG:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                num_classes=args.num_classes,
                large_feature_map=args.spg_large_feature_map,
                scale_in=p.scale_in
            )
        elif args.method == constants.METHOD_ADL:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                num_classes=args.num_classes,
                adl_drop_rate=args.adl_drop_rate,
                adl_drop_threshold=args.adl_drop_threshold,
                large_feature_map=args.adl_large_feature_map,
                scale_in=p.scale_in
            )

        elif args.method == constants.METHOD_TSCAM:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                num_classes=args.num_classes
            )
        elif args.method == constants.METHOD_SAT:
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                num_classes=args.num_classes,
                drop_rate=args.sat_drop_rate,
                drop_path_rate=args.sat_drop_path_rate
            )

        else:
            raise ValueError
    else:

        aux_params = get_aux_params(args)
        model = create_model(
            task=constants.STD_CL,
            arch=constants.STDCLASSIFIER,
            method='',
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=p.in_channels,
            encoder_depth=encoder_depth,
            scale_in=p.scale_in,
            aux_params=aux_params
        )

    DLLogger.log("PRETRAINED CLASSIFIER `{}` was created. "
                 "Nbr.params: {}".format(model, count_nb_params(model)))
    log = "Arch: {}\n" \
          "encoder_name: {}\n" \
          "encoder_weights: {}\n" \
          "classes: {}\n" \
          "aux_params: \n{}\n" \
          "scale_in: {}\n" \
          "freeze_cl: {}\n" \
          "img_range: {} \n" \
          "".format(p.arch, p.encoder_name,
                    encoder_weights, classes,
                    format_dict_2_str(
                        aux_params) if aux_params is not None else None,
                    p.scale_in, p.freeze_cl, args.img_range
                    )
    DLLogger.log(log)

    path_cl = args.model['folder_pre_trained_cl']
    assert path_cl not in [None, 'None', '']

    msg = "You have asked to set the classifier " \
          " from {} .... [OK]".format(path_cl)
    warnings.warn(msg)
    DLLogger.log(msg)

    if args.task == constants.NEGEV:
        cl_cp = args.negev_ptretrained_cl_cp
        std_cl_args = deepcopy(args)
        std_cl_args.task = constants.STD_CL
        tag = get_tag(std_cl_args, checkpoint_type=cl_cp)

    else:
        tag = get_tag(args)

    if path_cl.endswith(os.sep):
        source_tag = basename(path_cl[:-1])
    else:
        source_tag = basename(path_cl)

    assert tag == source_tag, f'{tag}, {source_tag}'

    if args.method in spec_mth:
        weights = torch.load(join(path_cl, 'model.pt'),
                             map_location=get_cpu_device())
        model.load_state_dict(weights, strict=True)
    else:
        encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                               map_location=get_cpu_device())
        model.encoder.super_load_state_dict(encoder_w, strict=True)

        header_w = torch.load(join(path_cl, 'classification_head.pt'),
                              map_location=get_cpu_device())
        model.classification_head.load_state_dict(header_w, strict=True)

    # if args.model['freeze_cl']:
    #     assert args.task == constants.F_CL
    #     assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']
    #
    #     model.freeze_classifier()
    #     model.assert_cl_is_frozen()

    model.eval()
    return model


def get_model(args, eval=False, eval_path_weights=''):
    """
    Returns the model to be trained.
    In the case of SFUDA, it returns the target model.
    """

    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL, constants.METHOD_TSCAM,
                constants.METHOD_SAT]
    method = ''
    support_background = args.model['support_background'],

    if args.task == constants.STD_CL:
        aux_params = None
        if args.method in spec_mth:

            if args.method == constants.METHOD_ACOL:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    num_classes=args.num_classes,
                    acol_drop_threshold=args.acol_drop_threshold,
                    large_feature_map=args.acol_large_feature_map,
                    scale_in=p.scale_in,
                    spatial_dropout=p.spatial_dropout
                )
            elif args.method == constants.METHOD_SPG:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    num_classes=args.num_classes,
                    large_feature_map=args.spg_large_feature_map,
                    scale_in=p.scale_in,
                    spatial_dropout=p.spatial_dropout
                )
            elif args.method == constants.METHOD_ADL:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    num_classes=args.num_classes,
                    adl_drop_rate=args.adl_drop_rate,
                    adl_drop_threshold=args.adl_drop_threshold,
                    large_feature_map=args.adl_large_feature_map,
                    scale_in=p.scale_in,
                    spatial_dropout=p.spatial_dropout
                )

            elif args.method == constants.METHOD_TSCAM:
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    num_classes=args.num_classes,
                    spatial_dropout=p.spatial_dropout
                )
            elif args.method == constants.METHOD_SAT:
                aux_params = None
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    num_classes=args.num_classes,
                    drop_rate=args.sat_drop_rate,
                    drop_path_rate=args.sat_drop_path_rate,
                    aux_params=aux_params,
                    pixel_wise_classification=args.pixel_wise_classification,
                    freeze_cl=p.freeze_cl
                )

            else:
                raise ValueError
        elif args.method == constants.METHOD_MAXMIN:

            assert p.spatial_dropout == 0.0, f"{p.spatial_dropout}: not " \
                                             f"supported"

            aux_params = get_aux_params(args)
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                encoder_depth=encoder_depth,
                scale_in=p.scale_in,
                aux_params=aux_params,
                w=args.maxmin_w,
                dataset_name=args.dataset,
                pixel_wise_classification=args.pixel_wise_classification,
                freeze_cl=p.freeze_cl
            )
            
        elif args.method == constants.METHOD_ENERGY:
                aux_params = get_aux_params(args)
                model = create_model(
                    task=args.task,
                    arch=p.arch,
                    method=args.method,
                    encoder_name=p.encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=p.in_channels,
                    encoder_depth=encoder_depth,
                    scale_in=p.scale_in,
                    spatial_dropout=p.spatial_dropout,
                    aux_params=aux_params,
                    pixel_wise_classification=args.pixel_wise_classification,
                    freeze_cl=p.freeze_cl,
                    num_classes=args.num_classes,
                    drop_rate=args.sat_drop_rate,
                    drop_path_rate=args.sat_drop_path_rate
                )
        else:
            aux_params = get_aux_params(args)
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                encoder_depth=encoder_depth,
                scale_in=p.scale_in,
                spatial_dropout=p.spatial_dropout,
                aux_params=aux_params
            )

    elif args.task == constants.F_CL:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            spatial_dropout=p.spatial_dropout,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )

    elif args.task == constants.NEGEV:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            spatial_dropout=p.spatial_dropout,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )

    elif args.task == constants.SEG:
        assert args.dataset in [constants.GLAS, constants.CAMELYON512]
        assert args.seg_mode == constants.BINARY_MODE
        assert classes == 2

        aux_params = None

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=method,
            encoder_name=p.encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            classes=classes)
    else:
        raise NotImplementedError

    DLLogger.log("`{}` was created. Nbr.params: {}".format(
        model,  count_nb_params(model)))
    log = "Arch: {}\n" \
          "task: {}\n" \
          "encoder_name: {}\n" \
          "encoder_weights: {}\n" \
          "classes: {}\n" \
          "aux_params: \n{}\n" \
          "scale_in: {}\n" \
          "freeze_cl: {}\n" \
          "im_rec: {}\n" \
          "img_range: {} \n" \
          "".format(p.arch, args.task, p.encoder_name,
                    encoder_weights, classes,
                    format_dict_2_str(
                        aux_params) if aux_params is not None else None,
                    p.scale_in, p.freeze_cl, args.im_rec, args.img_range
                    )
    DLLogger.log(log)
    DLLogger.log(model.get_info_nbr_params())


    path_cl = args.model['folder_pre_trained_cl']
    if path_cl not in [None, 'None']:
        msg = "You have asked to load a specific pre-trained " \
              "model from {} .... [OK]".format(path_cl)
        warnings.warn(msg)
        DLLogger.log(msg)
        if args.method == constants.METHOD_ENERGY and "deit" in args.model['encoder_name']:
                    weights = torch.load(join(path_cl, 'model.pt'),
                                     map_location=get_cpu_device())
                    model.load_state_dict(weights, strict=False)
        else:
            encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                   map_location=get_cpu_device())
            model.encoder.super_load_state_dict(encoder_w, strict=True)

            header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                  map_location=get_cpu_device())
            model.classification_head.load_state_dict(header_w, strict=True)

    path_file = args.model['path_pre_trained']
    if path_file not in [None, 'None']:
        msg = "You have asked to load a specific pre-trained " \
              "model from {} .... [OK]".format(path_file)
        warnings.warn(msg)
        DLLogger.log(msg)
        pre_tr_state = torch.load(path_file, map_location=get_cpu_device())
        model.load_state_dict(pre_tr_state, strict=args.model['strict'])
        

    if args.task in [constants.F_CL, constants.NEGEV]:
        path_cl = args.model['folder_pre_trained_cl']
        if path_cl not in [None, 'None', '']:
            assert args.task in [constants.F_CL, constants.NEGEV]

            msg = "You have asked to set the classifier's weights " \
                " from {} .... [OK]".format(path_cl)
            warnings.warn(msg)
            DLLogger.log(msg)

            if args.task == constants.NEGEV:
                cl_cp = args.negev_ptretrained_cl_cp
                std_cl_args = deepcopy(args)
                std_cl_args.task = constants.STD_CL
                tag = get_tag(std_cl_args, checkpoint_type=cl_cp)

            else:
                tag = get_tag(args)

            if path_cl.endswith(os.sep):
                source_tag = basename(path_cl[:-1])
            else:
                source_tag = basename(path_cl)

            assert tag == source_tag, f'{tag}, {source_tag}'

            if args.method in spec_mth:
                weights = torch.load(join(path_cl, 'model.pt'),
                                    map_location=get_cpu_device())
                model.load_state_dict(weights, strict=True)
            else:
                encoder_w = torch.load(join(path_cl, 'encoder.pt'),
                                    map_location=get_cpu_device())
                model.encoder.super_load_state_dict(encoder_w, strict=True)

                header_w = torch.load(join(path_cl, 'classification_head.pt'),
                                    map_location=get_cpu_device())
                model.classification_head.load_state_dict(header_w, strict=True)

    if args.model['freeze_cl'] and not eval:
        assert args.task in [constants.F_CL, constants.NEGEV] or args.method == constants.METHOD_ENERGY

        assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']

        model.freeze_classifier()
        model.assert_cl_is_frozen()

        if args.sf_uda:
            raise NotImplementedError  # todo

    # SFUDA
    model_src = None

    if args.sf_uda:
        model = sf_uda_load_set_source_weights(model, args)

        model_src = deepcopy(model)
        model_src.eval()
        freeze_all_params(model_src)

        if args.shot or args.faust or args.sfde or args.cdcl or args.esfda:  # shot/faust/sfde/cdcl methods
            model.train()
            model.freeze_cl_hypothesis()  # last linear weights + bias of
            # classifier. some wsol methods do not have a last linear
            # classifier: either simple fully conv layers, attention,
            # or no weights (simple max pooling for e.g.)

        elif args.adadsa:
            # Estimate BN stats over target trainset.
            bn_stats_estimator = adadsa.AdadsaEstimateTrgBnStats(
                args, deepcopy(model).cuda(args.c_cudaid))
            model_with_bn_trg_stats = bn_stats_estimator.estimate_bn_stats()

            # Fuse source and target BNs
            s_model = deepcopy(model)
            t_model = model_with_bn_trg_stats.to(get_cpu_device())

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

            # set only alpha as learnable param
            model = adadsa.adadsa_freeze_all_model_except_bn_a(model)

        elif args.sdda or args.nrc:
            model.train()  # adapt/nrc full model.
        else:  # todo
            raise NotImplementedError('add more SFUDA methods.')


    if eval:
        if os.path.isdir(eval_path_weights):
            path = eval_path_weights
        else:
            assert os.path.isdir(args.outd)
            tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
            path = join(args.outd, tag)
        cpu_device = get_cpu_device()

        if args.task == constants.STD_CL:
            if args.method in spec_mth:
                weights = torch.load(join(path, 'model.pt'),
                                     map_location=cpu_device)
                model.load_state_dict(weights, strict=True)
            else:
                weights = torch.load(join(path, 'encoder.pt'),
                                     map_location=cpu_device)
                model.encoder.super_load_state_dict(weights, strict=True)

                weights = torch.load(join(path, 'classification_head.pt'),
                                     map_location=cpu_device)
                model.classification_head.load_state_dict(weights, strict=True)

        elif args.task == constants.F_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)
            if model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=cpu_device)
                model.reconstruction_head.load_state_dict(weights, strict=True)

        elif args.task == constants.NEGEV:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'classification_head.pt'),
                                 map_location=cpu_device)
            model.classification_head.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)

        elif args.task == constants.SEG:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)
        else:
            raise NotImplementedError

        model.eval()

        msg = "EVAL-mode. Reset model weights to: {}".format(path)
        warnings.warn(msg)
        DLLogger.log(msg)


    return model, model_src


def sfuda_get_gan_sdda_model(args,
                             eval: bool = False,
                             eval_path_weights: str = ''
                             ):
    """
    Return the GAN (Generator and Discriminator) of the method SDDA for SFUDA.
    """
    assert args.sf_uda
    assert args.sdda
    assert args.sdda_gan_type in constants.GANS, f"{args.sdda_gan_type} |" \
                                                 f" {constants.GANS}"

    if args.sdda_gan_type == constants.CGAN_ORIGINAL:
        n_cls = args.num_classes
        latent_dim = args.sdda_gan_latent_dim
        img_shape = [3, args.sdda_gan_h, args.sdda_gan_w]  # RGB.

        gen = cgan.Generator(n_classes=n_cls,
                             latent_dim=latent_dim,
                             img_shape=img_shape
                             )
        disc = cgan.Discriminator(img_shape=img_shape)

        gen.train()
        disc.train()

        if eval:
            if os.path.isdir(eval_path_weights):
                path = eval_path_weights
            else:
                assert os.path.isdir(args.outd)
                tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
                path = join(args.outd, tag)

            cpu_device = get_cpu_device()

            params = torch.load(join(path, 'generator.pt'),
                                map_location=cpu_device)
            gen.load_state_dict(params, strict=True)

            params = torch.load(join(path, 'discriminator.pt'),
                                map_location=cpu_device)
            disc.load_state_dict(params, strict=True)

            gen.eval()
            disc.eval()


    else:
        raise NotImplementedError(args.sdda_gan_type)

    return gen, disc


def sfuda_get_domain_discriminator_sdda_model(args,
                                              features_dim: int,
                                              eval: bool = False,
                                              eval_path_weights: str = ''
                                              ):
    """
    Return the domain discriminator model of the method SDDA for SFUDA.
    """
    assert args.sf_uda
    assert args.sdda

    disc = uda_backprop.DomainDiscriminator(features_dim=features_dim)

    disc.train()

    if eval:
        if os.path.isdir(eval_path_weights):
            path = eval_path_weights
        else:
            assert os.path.isdir(args.outd)
            tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
            path = join(args.outd, tag)

        cpu_device = get_cpu_device()

        params = torch.load(join(path, 'uda_backprop_domain_discriminator.pt'),
                            map_location=cpu_device)
        disc.load_state_dict(params, strict=True)

        disc.eval()

    return disc


def sf_uda_load_set_source_weights(model, args: object):
    assert args.sf_uda
    assert args.task == constants.STD_CL, args.task  # todo: others.

    fd = args.sf_uda_source_folder
    assert os.path.isdir(fd), fd

    _args_trg = deepcopy(args)
    with open(join(fd, 'config_model.yaml'), 'r') as fx:
        args_src = yaml.load(fx, Loader=yaml.Loader)  # safe_load fails.
        args_src = Dict2Obj(args_src)

    # sanity check

    # convention: source wsol method == target wsol method. [not necessary]

    # requested source model matches the real source model
    assert _args_trg.sf_uda_source_encoder_name == args_src.model['encoder_name']
    assert _args_trg.sf_uda_source_wsol_method == args_src.method
    assert _args_trg.sf_uda_source_wsol_arch == args_src.model['arch']
    assert _args_trg.sf_uda_source_wsol_spatial_pooling == args_src.spatial_pooling

    # target model matches source model.
    assert _args_trg.model['encoder_name'] == args_src.model['encoder_name']
    assert _args_trg.method == args_src.method
    assert _args_trg.model['arch'] == args_src.model['arch']
    assert _args_trg.spatial_pooling == args_src.spatial_pooling

    assert _args_trg.sf_uda_source_ds == args_src.dataset
    assert _args_trg.num_classes == args_src.num_classes

    checkpoint_type = _args_trg.sf_uda_source_checkpoint_type
    assert checkpoint_type in [constants.BEST_LOC, constants.BEST_CL], checkpoint_type

    assert _args_trg.task == args_src.task, f"{_args_trg.task} | " \
                                            f"{args_src.task}"

    if args.task == constants.NEGEV:
        raise NotImplementedError  # todo
    else:
        tag = get_tag(args_src, checkpoint_type=checkpoint_type)

    path = fd
    cpu_device = get_cpu_device()
    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL, constants.METHOD_TSCAM,
                constants.METHOD_SAT]

    if args.task == constants.STD_CL:
        if args.method in spec_mth:
            weights = torch.load(join(path, 'model.pt'),
                                 map_location=get_cpu_device())
            model.load_state_dict(weights, strict=True)
        else:
            if args.method == constants.METHOD_ENERGY and   'deit' in args.model['encoder_name']:
                weights = torch.load(join(path, 'model.pt'),
                                 map_location=get_cpu_device())
                model.load_state_dict(weights, strict=True)
            else:
                weights = torch.load(join(path, 'encoder.pt'),
                                    map_location=cpu_device)
                model.encoder.super_load_state_dict(weights, strict=True)

                weights = torch.load(join(path, 'classification_head.pt'),
                                    map_location=cpu_device)
                model.classification_head.load_state_dict(weights, strict=True)
                if args.method == constants.METHOD_ENERGY:
                    weights = torch.load(join(path, 'pixel_wise_classification_head.pt'),
                                        map_location=cpu_device)
                    model.pixel_wise_classification_head.load_state_dict(weights, strict=True)

    elif args.task == constants.F_CL:
        weights = torch.load(join(path, 'encoder.pt'),
                             map_location=cpu_device)
        model.encoder.super_load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'decoder.pt'),
                             map_location=cpu_device)
        model.decoder.load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'segmentation_head.pt'),
                             map_location=cpu_device)
        model.segmentation_head.load_state_dict(weights, strict=True)
        if model.reconstruction_head is not None:
            weights = torch.load(join(path, 'reconstruction_head.pt'),
                                 map_location=cpu_device)
            model.reconstruction_head.load_state_dict(weights, strict=True)

    elif args.task == constants.NEGEV:
        weights = torch.load(join(path, 'encoder.pt'),
                             map_location=cpu_device)
        model.encoder.super_load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'decoder.pt'),
                             map_location=cpu_device)
        model.decoder.load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'classification_head.pt'),
                             map_location=cpu_device)
        model.classification_head.load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'segmentation_head.pt'),
                             map_location=cpu_device)
        model.segmentation_head.load_state_dict(weights, strict=True)

    elif args.task == constants.SEG:
        weights = torch.load(join(path, 'encoder.pt'),
                             map_location=cpu_device)
        model.encoder.super_load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'decoder.pt'),
                             map_location=cpu_device)
        model.decoder.load_state_dict(weights, strict=True)

        weights = torch.load(join(path, 'segmentation_head.pt'),
                             map_location=cpu_device)
        model.segmentation_head.load_state_dict(weights, strict=True)
    else:
        raise NotImplementedError

    msg = f"Loaded source weights (SFUDA) from: {fd}"
    warnings.warn(msg)
    DLLogger.log(fmsg(msg))

    return model


def standardize_optimizers_hparams(optm_dict: dict, initial: str):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'initial__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """
    assert isinstance(optm_dict, dict), type(optm_dict)
    new_optm_dict = deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith(initial):
            msg = f"'{k}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing...."
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            assert new_k not in new_optm_dict, new_k
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def _get_model_params_for_opt(args, model):
    hparams = deepcopy(args.optimizer)
    hparams = standardize_optimizers_hparams(hparams, 'opt')
    hparams = Dict2Obj(hparams)

    if args.task in [constants.F_CL, constants.SEG]:
        return [
            {'params': model.parameters(), 'lr': hparams.lr}
        ]

    spec_mth = [constants.METHOD_SPG, constants.METHOD_ACOL,
                constants.METHOD_ADL, constants.METHOD_TSCAM,
                constants.METHOD_SAT]

    sp_method = (args.task == constants.STD_CL) and (args.method in spec_mth)

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    architecture = args.model['encoder_name']
    assert architecture in constants.BACKBONES

    if args.method in [constants.METHOD_TSCAM, constants.METHOD_SAT, constants.METHOD_ENERGY]:
        if 'deit' in architecture:
            return [
                {'params': model.parameters(), 'lr': hparams.lr}
            ]
    
    if not sp_method:
        _FEATURE_PARAM_LAYER_PATTERNS = {
            'vgg': ['encoder.features.'],  # features
            'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
            'inception': ['encoder.Mixed', 'encoder.Conv2d_1',
                          'encoder.Conv2d_2',
                          'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
        }

   


    param_features = []
    param_classifiers = []

    def param_features_substring_list(arch):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if arch.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(arch))

    for name, parameter in model.named_parameters():

        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_features.append(parameter)
            elif architecture == constants.RESNET50:
                param_classifiers.append(parameter)
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_classifiers.append(parameter)
            elif architecture == constants.RESNET50:
                param_features.append(parameter)

    return [
            {'params': param_features, 'lr': hparams.lr},
            {'params': param_classifiers,
             'lr': hparams.lr * hparams.lr_classifier_ratio}
    ]


def build_optimizer_for_params(params: Iterable[object], hparams: object):
    """
    Builds an optimizer for the given params, and their hyper-paramerters.
    """

    if hparams.name_optimizer == constants.SGD:
        optimizer = SGD(params=params,
                        momentum=hparams.momentum,
                        dampening=hparams.dampening,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov
                        )

    elif hparams.name_optimizer == constants.ADAM:
        optimizer = Adam(params=params,
                         betas=(hparams.beta1, hparams.beta2),
                         eps=hparams.eps_adam,
                         weight_decay=hparams.weight_decay,
                         amsgrad=hparams.amsgrad)

    else:
        raise ValueError(f"Unsupported optimizer name "
                         f"`{hparams.name_optimizer}` ... [NOT OK]")

    if hparams.lr_scheduler:
        if hparams.name_lr_scheduler == constants.STEP:
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=hparams.step_size,
                                                  gamma=hparams.gamma,
                                                  last_epoch=hparams.last_epoch
                                                  )


        elif hparams.name_lr_scheduler == constants.COSINE:
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.t_max,
                eta_min=hparams.min_lr,
                last_epoch=hparams.last_epoch
            )

        elif hparams.name_lr_scheduler == constants.MYSTEP:
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer,
                step_size=hparams.step_size,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch,
                min_lr=hparams.min_lr)


        elif hparams.name_lr_scheduler == constants.MYCOSINE:
            lrate_scheduler = my_lr_scheduler.MyCosineLR(
                optimizer,
                coef=hparams.coef,
                max_epochs=hparams.max_epochs,
                min_lr=hparams.min_lr,
                last_epoch=hparams.last_epoch)


        elif hparams.name_lr_scheduler == constants.MULTISTEP:
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hparams.milestones,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch)

        else:
            raise ValueError(f"Unsupported LR scheduler "
                             f"`{hparams.name_lr_scheduler}` ... [NOT OK]")
    else:
        lrate_scheduler = None


    return optimizer, lrate_scheduler


def get_optimizer_of_model(args, model):
    """
    Get the optimizer of the MAIN model.
    """
    hparams = deepcopy(args.optimizer)
    hparams = standardize_optimizers_hparams(hparams, 'opt')
    hparams = Dict2Obj(hparams)

    params = _get_model_params_for_opt(args, model)

    optimizer, lrate_scheduler = build_optimizer_for_params(params, hparams)

    return optimizer, lrate_scheduler


def get_optimizer_for_params(args_holder: dict,
                             params: Iterable[object],
                             initial: str
                             ):
    """
    Get optimizer for a set of parameters. Hyper-parameters are in args_holder.
    :param args_holder: dict containing hyper-parameters.
    :param params: list of params.
    :param initial: str. initial to extract the hyper-parameters.
    :return:
    """
    hparams = deepcopy(args_holder)
    hparams = standardize_optimizers_hparams(hparams, initial)
    hparams = Dict2Obj(hparams)

    _params = [
        {'params': params, 'lr': hparams.lr}
    ]

    optimizer, lrate_scheduler = build_optimizer_for_params(_params, hparams)

    return optimizer, lrate_scheduler