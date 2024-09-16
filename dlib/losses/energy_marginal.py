from torch import nn
#from utils.utils import to_cuda
import torch
import torch.nn.functional as F


class Energy_Marginal(torch.nn.Module):
    def forward(self, energy_target, energy_samples):
        fp = energy_target.mean()
        fq = energy_samples.mean()
        loss_p_x = -(fp - fq)

        return loss_p_x