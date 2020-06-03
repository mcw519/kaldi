import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_aed_model(input_dim, output_dim):
    model = AedModel(input_dim=40,
                     output_dim=40)

    return model

class AedModel(nn.Module):

    def __init__(self,
                 input_dim=40,
                 output_dim=40):
        super().__init__()


        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=20, kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=20, affine=False),
            nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=10, affine=False),
            nn.Conv1d(in_channels=10, out_channels=5, kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=5, affine=False),
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=5, affine=False),
            nn.Conv1d(in_channels=5, out_channels=5, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=5, affine=False)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=5, out_channels=10, kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=10, affine=False),
            nn.ConvTranspose1d(in_channels=10, out_channels=20, kernel_size=2),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=20, affine=False),
            nn.ConvTranspose1d(in_channels=20, out_channels=40, kernel_size=2),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=40, affine=False)
        )

    def forward(self, feat, feat_len_list):
        '''
        Args:
            feat: a 3-D tensor of shape [batch_size, seq_len, feat_dim]
            feat_len_list: feat length of each utterance before padding

        Returns:
            a 3-D tensor of shape [batch_size, seq_len, output_dim]
            It is the output of `nn.Linear`. That is, **NO** log_softmax
            is applied to the output.
        '''
        
        x = feat
        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        x = x.permute(0, 2, 1)

        # at this point, x is [batch_size, feat_dim, seq_len] = [N, C, T]

        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        return x, feat_len_list

def _test_aed_model():
    input_dim = 40
    output_dim = 40
    model = AedModel(input_dim=input_dim,
                     output_dim=output_dim)

    feat1 = torch.randn((6, input_dim))
    feat2 = torch.randn((8, input_dim))

    from torch.nn.utils.rnn import pad_sequence
    feat = pad_sequence([feat1, feat2], batch_first=True)

    feat_len_list = [6, 8]

    x, _ = model(feat, feat_len_list)

    assert x.shape == torch.Size([2, 8, output_dim])

if __name__ == '__main__':
    _test_aed_model()
