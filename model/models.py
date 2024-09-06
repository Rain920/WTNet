import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from model.layers import BiLSTM, BiGRU


class WTNet_cl(nn.Module):
    def __init__(self):
        super(WTNet_cl, self).__init__()
        self.filter_conv = self._init_filter_conv()

        # self.gru = BiGRU(input_size=256, output_size=256, hidden_size=256, num_layers=1, batch_first=True, return_sequence=True)
        self.bilstm = BiLSTM(input_size=256, output_size=256, hidden_size=256, num_layers=1, batch_first=True, return_sequence=True)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256*118, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.flatten = nn.Flatten()

        self.temperature1 = 0.9
        self.temperature2 = 0.9
        self.lambda1 = 0.5
        self.lambda2 = 0.5

    
    def _init_filter_conv(self):
        # 16, 16, 32, 32, 64, 64, 128, 128, and 256
        kernel_size = 7
        padding_size = (kernel_size - 1) // 2
        kernel_size1 = 3
        padding_size1 = (kernel_size1 - 1) // 2
        stride = 2
        stride1 = 5
        conv_layers = [
            nn.Conv1d(1, 16, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding_size),
            nn.ReLU()
        ]
        filter_conv = nn.Sequential(*conv_layers)
        for m in filter_conv:
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
        return filter_conv
    
    def nt_xent_loss(self, features, labels):
        batch_size = features.shape[0]
        features = self.flatten(features)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # build mask according to labels, positive: 1, negative: 0
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        # mask diagonal
        mask = mask.fill_diagonal_(0)

        # Compute the numerator and denominator of the NT-Xent loss
        numerator = torch.exp(similarity_matrix / self.temperature1) * mask
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature1), dim=1, keepdim=True) \
                    - torch.exp(similarity_matrix.diagonal() / self.temperature1).unsqueeze(1)

        ntx_loss = -torch.log(numerator.sum(dim=1) / denominator)
        ntx_loss = ntx_loss.mean()

        return ntx_loss
    
    def nce_loss(self, features, labels):
        batch_size = features.shape[0]

        features = self.flatten(features)

        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute logits (cosine similarity matrix scaled by temperature)
        logits = torch.matmul(features, features.T) / self.temperature2
        
        # Create mask to remove self-similarity
        mask = torch.eye(batch_size, device=features.device).bool()
        
        # Apply mask to logits to remove self-similarity
        logits.masked_fill_(mask, float('-inf'))
        
        # Create positive and negative labels
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Compute the probability of positive samples
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
        
        pos_logits = exp_logits * positive_mask
        neg_logits = exp_logits * negative_mask
        
        pos_prob = pos_logits.sum(dim=1) / sum_exp_logits.squeeze()
        neg_prob = neg_logits.sum(dim=1) / sum_exp_logits.squeeze()
        
        # Compute NCE loss
        nce_loss = -torch.log(pos_prob) - torch.log(1 - neg_prob)
        nce_loss = nce_loss.mean()

        return nce_loss

    def forward(self, x, y=None):
        contrastive_loss = 0
        x = self.filter_conv(x)
        x = x.permute(0, 2, 1)
        x = self.bilstm(x)
        if y is not None:
            ntx_loss = self.nt_xent_loss(x, y)
            nce_loss = self.nce_loss(x, y)
            contrastive_loss = self.lambda1 * ntx_loss + self.lambda2 * nce_loss

        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)

        return x, contrastive_loss