import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = features.device

        # Normalize the vectors
        features = F.normalize(features, p=2, dim=1)

        # Extend the dimensions of the labels
        #labels = labels[:, None]
        labels_expanded_1 = labels.unsqueeze(2)  # Shape : [32, 224, 1, 224]
        labels_expanded_2 = labels.unsqueeze(3)  # Shape : [32, 224, 224, 1]

        valid_mask = labels != -255

        valid_mask_1 = valid_mask.unsqueeze(2)  # Forme : [32, 224, 1, 224]
        valid_mask_2 = valid_mask.unsqueeze(3)  # Forme : [32, 224, 224, 1]


        # Mask for positive pairs
        mask = torch.eq(labels_expanded_1, labels_expanded_2) & valid_mask_1 & valid_mask_2
        #mask = torch.eq(labels_expanded_1, labels_expanded_2).bool().to(device)

        # Identity matrix to exclude self-pairs
        batch_size, height, width = labels.shape

        eye = torch.eye(height, width, device=labels.device).bool()

        #eye = torch.eye(mask.shape[1], device=device).bool()

        # Masks for positive and negative pairs
        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()

        # Dot product between vectors
        dot_prod = torch.matmul(features, features.t())

        # Mean of dot products for positive and negative pairs
        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        # Compute the loss
        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss