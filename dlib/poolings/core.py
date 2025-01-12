import sys
from os.path import dirname, abspath

import re
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

__all__ = ['GAP', 'WGAP', 'MaxPool', 'LogSumExpPool', 'PRM']


class _BasicPooler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 classes: int,
                 support_background: bool = False,
                 r: float = 10.,
                 modalities: int = 5,
                 kmax: float = 0.5,
                 kmin: float = None,
                 alpha: float = 0.6,
                 dropout: float = 0.0,
                 mid_channels: int = 128,
                 gated: bool = False,
                 prm_ks: int = 3,
                 prm_st: int = 1,
                 pixel_wise_classification: bool = False,
                 freeze_cl: bool = False,
                 batch_norm: bool = False, 
                 multiple_layer: bool = False, 
                 one_layer: bool = False,
                 anchors_ortogonal: bool = False, 
                 detach_pixel_classifier: bool = False
                 ):
        super(_BasicPooler, self).__init__()

        self.cams = None
        self.in_channels = in_channels
        self.classes = classes
        self.support_background = support_background

        # logsumexp
        self.r = r
        # wildcat
        self.modalities = modalities
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha
        self.dropout = dropout

        # mil
        self.mid_channels = mid_channels
        self.gated = gated

        # prm
        assert isinstance(prm_ks, int)
        assert prm_ks > 0
        assert isinstance(prm_st, int)
        assert prm_st > 0
        self.prm_ks = prm_ks
        self.prm_st = prm_st

        self.name = 'null-name'

        # SFUDA
        self.lin_ft = None  # linear features of the last layer in net to
        # produce image global class logits.

    def flush(self):
        self.lin_ft = None

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        pass

    @staticmethod
    def freeze_part(part):

        for module in (part.modules()):

            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    @property
    def builtin_cam(self):
        return True

    def assert_x(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

    def correct_cl_logits(self, logits):
        if self.support_background:
            return logits[:, 1:]
        else:
            return logits

    def get_nbr_params(self):
        return sum([p.numel() for p in self.parameters()])

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={})'.format(
            self.__class__.__name__, self.in_channels, self.classes,
            self.support_background)


class GAP(_BasicPooler):
    """ https://arxiv.org/pdf/1312.4400.pdf """
    def __init__(self, **kwargs):
        super(GAP, self).__init__(**kwargs)
        self.name = 'GAP'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        ft = self.pool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz

        out = self.conv(x)
        self.cams = out.detach()
        logits = self.pool(out).flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits


class WGAP(_BasicPooler):
    """ https://arxiv.org/pdf/1512.04150.pdf """
    def __init__(self, **kwargs):
        super(WGAP, self).__init__(**kwargs)
        self.name = 'WGAP'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, classes)

    @property
    def builtin_cam(self):
        return False
    
    def get_linear_weights(self):
        if self.support_background:
            return self.fc.weight[1:]
        else:
            return self.fc.weight

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        self.lin_ft = pre_logit  # bsz, sz

        logits = self.fc(pre_logit)

        logits = self.correct_cl_logits(logits)

        return logits


class MaxPool(_BasicPooler):
    def __init__(self, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.name = 'MaxPool'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.conv)  # warning: not linear cl.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        ft = self.avg_pool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz

        out = self.conv(x)
        self.cams = out.detach()
        logits = self.pool(out).flatten(1)

        logits = self.correct_cl_logits(logits)
        return logits


class LogSumExpPool(_BasicPooler):
    """ https://arxiv.org/pdf/1411.6228.pdf """
    def __init__(self, **kwargs):
        super(LogSumExpPool, self).__init__(**kwargs)
        self.name = 'LogSumExpPool'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.conv)  # warning: not linear cl.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        ft = self.avg_pool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz

        out = self.conv(x)
        self.cams = out.detach()
        out = self.avgpool((self.r * out).exp()).log() * (1/self.r)

        logits = out.flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={}, ' \
               'r={})'.format(self.__class__.__name__, self.in_channels,
                              self.classes, self.support_background, self.r)


class PRM(_BasicPooler):
    def __init__(self, **kwargs):
        super(PRM, self).__init__(**kwargs)
        self.name = 'PRM'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=self.prm_ks, stride=self.prm_st)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.conv)  # warning: not linear cl.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.assert_x(x)

        ft = self.pool(x)
        ft = ft.reshape(ft.size(0), -1)
        self.lin_ft = ft  # bsz, sz

        out = self.conv(x)
        self.cams = out.detach()

        out = self.maxpool(out)

        logits = self.pool(out).flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits

    def __repr__(self):
        return '{}(kernel size={}, stride={}, support_background={}, ' \
               ')'.format(self.__class__.__name__, self.prm_ks,
                          self.prm_st, self.support_background)


class PixelWise(_BasicPooler):
    def __init__(self, **kwargs):
        super(PixelWise, self).__init__(**kwargs)
        self.name = 'PixelWise'
        self.batch_norm = kwargs['batch_norm']
        self.multiple_layer = kwargs['multiple_layer']
        self.one_layer = kwargs['one_layer']
        self.anchors_ortogonal = kwargs['anchors_ortogonal']
        self.detach_pixel_classifier = kwargs['detach_pixel_classifier']
        classes = self.classes
        #if self.support_background:
        #    classes = classes + 1

        #self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
        #                      kernel_size=1)

        # self.conv1 = nn.Conv2d(self.in_channels, 1024, kernel_size=1)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.relu2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(512, 256, kernel_size=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(256, classes, kernel_size=1)


        mid_features1 = self.in_channels // 2
        mid_features2 = mid_features1 // 2
        mid_features3 = mid_features2 // 2

        if self.multiple_layer:
            self.conv1 = self._make_layer(self.in_channels, mid_features1)
            self.conv2 = self._make_layer(mid_features1, mid_features2)
            self.conv3 = self._make_layer(mid_features2, mid_features3)
            self.conv4 = nn.Conv2d(mid_features3, classes, kernel_size=1)
        elif self.one_layer:
            self.conv3 = self._make_layer(self.in_channels, mid_features2)
            self.conv4 = nn.Conv2d(mid_features2, classes, kernel_size=1)
        else:
            if self.batch_norm:
                self.bn = nn.BatchNorm2d(self.in_channels)
                self.conv4 = nn.Conv2d(self.in_channels, classes, kernel_size=1)
            else:
                self.conv4 = nn.Conv2d(self.in_channels, classes, kernel_size=1)

        if self.anchors_ortogonal:
            vectors = self.generate_orthogonal_vectors(self.in_channels)
            self.update_weights_with_vectors(vectors)
            self.freeze_classifier()

        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #self.layer1 = self._make_layer(self.in_channels, mid_features1)
        #self.layer2 = self._make_layer(mid_features1, mid_features2)
        #self.layer3 = self._make_layer(mid_features2, mid_features3)
        #self.conv4 = nn.Conv2d(mid_features3, classes, kernel_size=1)

        
    def update_weights_with_vectors(self, vectors):

        vectors = torch.tensor(vectors, dtype=self.conv4.weight.dtype, device=self.conv4.weight.device)
        
        # Reshape ou slice les vecteurs pour correspondre à la forme des poids
        weight_shape = self.conv4.weight.shape  # (out_channels, in_channels, kernel_size, kernel_size)
        in_channels = weight_shape[1]
        out_channels = weight_shape[0]

        if vectors.shape[0] != in_channels or vectors.shape[1] != out_channels:
            raise ValueError("La taille des vecteurs ne correspond pas aux dimensions des poids.")

        # Adapt the weights from the vectors
        new_weights = vectors.view(*weight_shape)  

        # Update the weights of the layer
        with torch.no_grad():
            self.conv4.weight.copy_(new_weights)


    def generate_orthogonal_vectors(self, size_features):
        matrix = np.random.randn(size_features, self.classes)
        # QR decomposition
        q, r = np.linalg.qr(matrix)
        # return ortogonal vectors
        return q 


    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            #nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def freeze_cl_hypothesis(self):
        # SFUDA: freeze the last linear weights + bias of the classifier
        self.freeze_part(self.conv)  
        
    def freeze_classifier(self):
        if self.multiple_layer:
            layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        elif self.one_layer:
            layers =  [self.conv3, self.conv4]
        else:
            layers = [self.conv4]

        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor, return_cams:bool=True) -> torch.Tensor:
        self.assert_x(x)

        if self.detach_pixel_classifier:
            x = x.detach()
            
        #out1 = self.relu1(self.bn1(self.conv1(x)))
        #out2 = self.relu2(self.bn2(self.conv2(out1)))
        #out3 = self.relu3(self.bn3(self.conv3(out2)))
        #x = self.upsample(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # last_features = self.layer3(x)
        #logits = x
        #logits = x
        if self.multiple_layer:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            logits = self.conv4(x)
            return logits, logits
        
        if self.one_layer:
            x = self.conv3(x)
            logits = self.conv4(x)
            return logits, logits
        

        if self.batch_norm:
            x = self.bn(x)
        
        logits = self.conv4(x)
        # if return_cams:
        #     self.cams = logits
        return logits, logits
    

if __name__ == '__main__':
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    b, c, h, w = 3, 1024, 8, 8
    classes = 5
    x = torch.rand(b, c, h, w).to(DEVICE)

    for support_background in [True, False]:
        for cl in [GAP, WGAP, MaxPool, LogSumExpPool, PRM]:
            instance = cl(in_channels=c, classes=classes,
                          support_background=support_background)
            instance.to(DEVICE)
            announce_msg('TEsting {}'.format(instance))
            out = instance(x)
            if instance.builtin_cam:
                print('x: {}, cam: {}, logitcl shape: {}, logits: {}'.format(
                    x.shape, instance.cams.shape, out.shape, out))
            else:
                print('x: {}, logitcl shape: {}, logits: {}'.format(
                    x.shape, out.shape, out))
