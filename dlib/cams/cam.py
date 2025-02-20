# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import sys
from os.path import dirname, abspath, join

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger

from dlib.cams.core import _CAM
from dlib.cams.core import locate_linear_layer


""" Source: https://github.com/ml-edu/torch-cam """


__all__ = ['CAM', 'ScoreCAM', 'SSCAM', 'ISCAM']


class CAM(_CAM):
    """Implements a class activation map extractor as described in
     `"Learning Deep Features for Discriminative
      Localization" <https://arxiv.org/pdf/1512.04150.pdf>`_.

    The Class Activation Map (CAM) is defined for image classification models
    that have global pooling at the end of the visual feature extraction
    block. The localization map is computed as follows:

    .. math::
        L^{(c)}_{CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}
         A_k(x, y)\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`,
    and :math:`w_k^{(c)}` is the weight corresponding to class :math:`c` for
    unit :math:`k` in the fully
    connected layer..

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import CAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = CAM(model, 'layer4', 'fc')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: name of the target layer
        fc_layer: name of the fully CONNECTED layer
        input_shape: shape of the expected input tensor excluding the batch
          dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        fc_layer: Optional[str] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, input_shape)

        # If the layer is not specified, try automatic resolution
        if fc_layer is None:
            fc_layer = locate_linear_layer(model)
            # Warn the user of the choice
            if isinstance(fc_layer, str):
                DLLogger.log(f"no value was provided for `fc_layer`, thus set "
                             f"to '{fc_layer}'.")
            else:
                raise ValueError("unable to resolve `fc_layer` automatically,"
                                 " please specify its value.")
        # Softmax weight
        self._fc_weights = self.submodule_dict[fc_layer].weight.data
        # squeeze to accomodate replacement by Conv1x1
        if self._fc_weights.ndim > 2:
            self._fc_weights = self._fc_weights.view(
                *self._fc_weights.shape[:2])

    def _get_weights(self, class_idx: int,
                     scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        # Take the FC weights of the target class
        return self._fc_weights[class_idx, :]


class ScoreCAM(_CAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks"
    <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}
        A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target
    layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for
    class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m
        U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and
    :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: name of the target layer
        batch_size: batch size used to forward masked inputs
        input_shape: shape of the expected input tensor excluding the batch
        dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        batch_size: int = 32,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, input_shape)

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(
            self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _store_input(self, module: nn.Module, input: Tensor) -> None:
        """Store model input tensor"""

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx: int,
                     scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        self.hook_a: Tensor
        upsampled_a = self._normalize(self.hook_a, self.hook_a.ndim - 2)

        # Upsample it to input_size
        # 1 * O * M * N
        spatial_dims = self._input.ndim - 2
        interpolation_mode = 'bilinear' if spatial_dims == 2 else 'trilinear' if spatial_dims == 3 else 'nearest'
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[2:],
                                    mode=interpolation_mode,
                                    align_corners=False)

        # Use it as a mask
        # O * I * H * W
        masked_input = upsampled_a.squeeze(0).unsqueeze(1) * self._input

        # Initialize weights
        weights = torch.zeros(masked_input.shape[0],
                              dtype=masked_input.dtype,
                              device=masked_input.device)

        # Disable hook updates
        self._hooks_enabled = False
        # Switch to eval
        origin_mode = self.model.training
        self.model.eval()
        # Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(weights.shape[0] / self.bs)):

            selection_slice = slice(idx * self.bs,
                                    min((idx + 1) * self.bs, weights.shape[0]))
            with torch.no_grad():
                # Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(
                    self.model(masked_input[selection_slice]), dim=1)[
                                           :, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True
        # Put back the model in the correct mode
        self.model.training = origin_mode

        return weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.bs})"


class SSCAM(ScoreCAM):
    """Implements a class activation map extractor as described in
    `"SS-CAM: Smoothed Score-CAM for
    Sharper Visual Feature Localization"
    <https://arxiv.org/pdf/2006.14255.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{SS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}
        A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{N} \\sum\\limits_1^N softmax(Y^{(c)}(M_k) -
        Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer
    of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for
    class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m
        U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U`
    is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that
    follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import SSCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SSCAM(model, 'layer4')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: name of the target layer
        batch_size: batch size used to forward masked inputs
        num_samples: number of noisy samples used for weight computation
        std: standard deviation of the noise added to the normalized activation
        input_shape: shape of the expected input tensor excluding the batch
           dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        batch_size: int = 32,
        num_samples: int = 35,
        std: float = 2.0,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, batch_size, input_shape)

        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def _get_weights(self, class_idx: int,
                     scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        self.hook_a: Tensor
        upsampled_a = self._normalize(self.hook_a, self.hook_a.ndim - 2)

        # Upsample it to input_size
        # 1 * O * M * N
        spatial_dims = self._input.ndim - 2
        interpolation_mode = 'bilinear' if spatial_dims == 2 else 'trilinear' if spatial_dims == 3 else 'nearest'
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[2:],
                                    mode=interpolation_mode,
                                    align_corners=False)

        # Use it as a mask
        # O * I * H * W
        upsampled_a = upsampled_a.squeeze(0).unsqueeze(1)

        # Initialize weights
        weights = torch.zeros(upsampled_a.shape[0],
                              dtype=upsampled_a.dtype,
                              device=upsampled_a.device)

        # Disable hook updates
        self._hooks_enabled = False
        # Switch to eval
        origin_mode = self.model.training
        self.model.eval()

        for _idx in range(self.num_samples):
            noisy_m = self._input * (
                    upsampled_a + self._distrib.sample(
                     self._input.size()).to(device=self._input.device))

            # Process by chunk (GPU RAM limitation)
            for idx in range(math.ceil(weights.shape[0] / self.bs)):

                selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs,
                                                           weights.shape[0]))
                with torch.no_grad():
                    # Get the softmax probabilities of the target class
                    weights[selection_slice] += F.softmax(
                        self.model(noisy_m[selection_slice]), dim=1)[
                                                :, class_idx]

        weights.div_(self.num_samples)

        # Reenable hook updates
        self._hooks_enabled = True
        # Put back the model in the correct mode
        self.model.training = origin_mode

        return weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.bs}, " \
               f"num_samples={self.num_samples}, std={self.std})"


class ISCAM(ScoreCAM):
    """Implements a class activation map extractor as described in
    `"IS-CAM: Integrated Score-CAM for axiomatic-based
    explanations" <https://arxiv.org/pdf/2010.03023.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{ISS-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)}
        A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^N \\frac{i}{N} softmax(Y^{(c)}(M_k) -
        Y^{(c)}(X_b))

    where :math:`N` is the number of samples used to smooth the weights,
    :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer
    of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for
    class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\Bigg(\\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m
        U(A_m) - \\min\\limits_m  U(A_m)} +
        \\delta\\Bigg) \\odot X

    where :math:`\\odot` refers to the element-wise multiplication, :math:`U`
    is the upsampling operation,
    :math:`\\delta \\sim \\mathcal{N}(0, \\sigma^2)` is the random noise that
    follows a 0-mean gaussian distribution
    with a standard deviation of :math:`\\sigma`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from dlib.cams import ISCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ISCAM(model, 'layer4')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: name of the target layer
        batch_size: batch size used to forward masked inputs
        num_samples: number of noisy samples used for weight computation
        input_shape: shape of the expected input tensor excluding the batch
         dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        batch_size: int = 32,
        num_samples: int = 10,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, batch_size, input_shape)

        self.num_samples = num_samples

    def _get_weights(self, class_idx: int,
                     scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        self.hook_a: Tensor
        upsampled_a = self._normalize(self.hook_a, self.hook_a.ndim - 2)

        # Upsample it to input_size
        # 1 * O * M * N
        spatial_dims = self._input.ndim - 2
        interpolation_mode = 'bilinear' if spatial_dims == 2 else 'trilinear' if spatial_dims == 3 else 'nearest'
        upsampled_a = F.interpolate(upsampled_a,
                                    self._input.shape[2:],
                                    mode=interpolation_mode,
                                    align_corners=False)

        # Use it as a mask
        # O * I * H * W
        upsampled_a = upsampled_a.squeeze(0).unsqueeze(1)

        # Initialize weights
        weights = torch.zeros(upsampled_a.shape[0],
                              dtype=upsampled_a.dtype,
                              device=upsampled_a.device)

        # Disable hook updates
        self._hooks_enabled = False
        fmap = torch.zeros((upsampled_a.shape[0], *self._input.shape[1:]),
                           dtype=upsampled_a.dtype, device=upsampled_a.device)
        # Switch to eval
        origin_mode = self.model.training
        self.model.eval()

        for _idx in range(self.num_samples):
            fmap += (_idx + 1) / self.num_samples * self._input * upsampled_a

            # Process by chunk (GPU RAM limitation)
            for idx in range(math.ceil(weights.shape[0] / self.bs)):

                selection_slice = slice(idx * self.bs, min((idx + 1) * self.bs,
                                                           weights.shape[0]))
                with torch.no_grad():
                    # Get the softmax probabilities of the target class
                    weights[selection_slice] += F.softmax(
                        self.model(fmap[selection_slice]), dim=1)[:, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True
        # Put back the model in the correct mode
        self.model.training = origin_mode

        return weights


if __name__ == "__main__":
    import datetime as dt
    import os
    import subprocess

    from torch.cuda.amp import autocast

    from torchvision.models import resnet50
    from torchvision.models import vgg16
    from torchvision.models import inception_v3

    from dlib.utils.reproducibility import set_seed
    from dlib.dllogger import ArbJSONStreamBackend
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend
    from dlib.dllogger import ArbTextStreamBackend

    import dlib
    from dlib.configure import constants
    from dlib import create_model
    from dlib.utils.shared import fmsg

    amp = True

    outd = join(root_dir, 'data/debug/cams')
    if not os.path.isdir(outd):
        os.makedirs(outd)

    exp_id = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
    log_backends = [ArbJSONStreamBackend(
        Verbosity.VERBOSE, join(outd, "log-cam-{}.json".format(exp_id))),
                    ArbTextStreamBackend(
                        Verbosity.VERBOSE,
                        join(outd, "log-cam-{}.txt".format(exp_id))),
                    ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends,master_pid=0)

    model_names = {resnet50: 'resnet50',
                   vgg16: 'vgg16',
                   inception_v3: 'inception_v3'}

    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    txt = subprocess.run(
        ['nvidia-smi', '--list-gpus'],
        stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    try:
        tag = txt[int(cuda)]
    except IndexError:
        tag = 'GPU'
    DLLogger.log('Device: {}'.format(tag))
    DLLogger.log('AMP: {}'.format(amp))

    def test_torchvision_models(modelx=resnet50):
        assert modelx == resnet50

        set_seed(0)
        b, c, h, w = 1, 3, 224, 224
        x = torch.rand(b, c, h, w).to(DEVICE)
        DLLogger.log('x input shape: {}'.format(x.shape))
        DLLogger.log('NBR CLASSES: {}'.format(1000))

        # test original.
        model = modelx(pretrained=True).eval()
        mname = model_names[modelx]
        DLLogger.log(fmsg('TEST torchvision models: {}'.format(mname)))

        model.to(DEVICE)
        target_layer = None
        methods = [CAM, ScoreCAM, SSCAM, ISCAM]

        for method in methods:
            if method == CAM:
                cam = method(model, target_layer=target_layer,
                             fc_layer=None)
            else:
                cam = method(model, target_layer=target_layer)
            DLLogger.log(fmsg('Testing: {} [{}]'.format(cam, mname)))
            with torch.no_grad():
                scores = model(x)
                DLLogger.log('cl scores shape: {}'.format(scores.shape))
            t0 = dt.datetime.now()
            pooled_cam = cam(class_idx=100)
            DLLogger.log('x: {}, cam: {}'.format(x.shape, pooled_cam.shape))
            DLLogger.log(fmsg('time: {}'.format(dt.datetime.now() - t0)))
            DLLogger.flush()


    def test_dlib_models(encoder_name):
        set_seed(0)
        b, c, h, w = 1, 3, 224, 224
        x = torch.rand(b, c, h, w).to(DEVICE)
        nbr_classes = 200
        class_idx = 88
        DLLogger.log('x input shape: {}'.format(x.shape))
        DLLogger.log('NBR CLASSES: {}'.format(nbr_classes))

        if encoder_name in [constants.VGG16]:
            vgg_encoders = dlib.encoders.vgg_encoders
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
            decoder_channels = (256, 128, 64)
        else:
            encoder_depth = 5
            decoder_channels = (256, 128, 64, 32, 16)

        support_background = True
        aux_params = {
            "pooling_head": constants.WGAP,
            "classes": nbr_classes,
            "modalities": 5,
            "kmax": 0.6,
            "kmin": 0.1,
            "alpha": 0.6,
            "dropout": 0.,
            "support_background": support_background
        }
        model = create_model(
            task=constants.STD_CL,
            arch=constants.STDCLASSIFIER,
            method=constants.METHOD_CAM,
            encoder_name=encoder_name,
            encoder_weights=constants.IMAGENET,
            in_channels=3,
            encoder_depth=encoder_depth,
            scale_in=1.,
            aux_params=aux_params
        )
        model.to(DEVICE)
        model.eval()

        DLLogger.log(fmsg('TEST DLIB models: {}'.format(encoder_name)))
        DLLogger.log(fmsg('Model log: {}'.format(model.get_info_nbr_params())))

        target_layer = constants.TRG_LAYERS[encoder_name]
        fc_layer = constants.FC_LAYERS[encoder_name]

        methods = [CAM, CAM, ScoreCAM, SSCAM, ISCAM]
        for method in methods:
            if method == CAM:
                cam = method(model, target_layer=target_layer,
                             fc_layer=fc_layer)
            else:
                cam = method(model, target_layer=target_layer)

            DLLogger.log(fmsg('Testing: {} [{}]'.format(cam, encoder_name)))
            # with torch.no_grad():
            scores = model(x)
            DLLogger.log("cl scores: {}".format(scores.shape))
            t0 = dt.datetime.now()
            with autocast(enabled=amp):
                pooled_cam = cam(class_idx=class_idx)

            if amp:
                pooled_cam = pooled_cam.float()

            if pooled_cam.shape != (h, w):
                tx = dt.datetime.now()
                full_cam = F.interpolate(
                        input=pooled_cam.unsqueeze(0).unsqueeze(0),
                        size=[h, w],
                        mode='bilinear',
                        align_corners=True)
            DLLogger.log('x: {}, cam: {}'.format(x.shape, pooled_cam.shape))
            DLLogger.log(fmsg('time: {} [{} {}]'.format(
                dt.datetime.now() - t0, cam, encoder_name)))
            DLLogger.flush()


    # test_torchvision_models()

    encoders = [constants.RESNET50, constants.VGG16, constants.INCEPTIONV3]
    for encoder in encoders:
        test_dlib_models(encoder)
        sys.exit()


