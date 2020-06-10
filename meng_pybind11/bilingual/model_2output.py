#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Copyright 2020 Author: Meng Wu
# Apache 2.0

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import load_lda_mat
from tdnnf_layer import FactorizedTDNN
from tdnnf_layer import OrthonormalLinear
from tdnnf_layer import PrefinalLayer
from tdnnf_layer import TDNN
from trick_layer import pdf_select, EntropyFunction

def get_chain_model(feat_dim,
                    output_dim,
                    ivector_dim,
                    hidden_dim,
                    bottleneck_dim,
                    prefinal_bottleneck_dim,
                    kernel_size_list,
                    subsampling_factor_list,
                    lda_mat_filename=None):
    model = ChainModel(feat_dim=feat_dim,
                       output_dim=output_dim,
                       ivector_dim=ivector_dim,
                       lda_mat_filename=lda_mat_filename,
                       hidden_dim=hidden_dim,
                       bottleneck_dim=bottleneck_dim,
                       prefinal_bottleneck_dim=prefinal_bottleneck_dim,
                       kernel_size_list=kernel_size_list,
                       subsampling_factor_list=subsampling_factor_list)
    return model


'''
input dim=43 name=input

# please note that it is important to have input layer with the name=input
# as the layer immediately preceding the fixed-affine-layer to enable
# the use of short notation for the descriptor
fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=exp/chain_cleaned_1c/tdnn1c_sp/configs/lda.mat

# the first splicing is moved before the lda layer, so no splicing here
relu-batchnorm-dropout-layer name=tdnn1 l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true dim=1024
tdnnf-layer name=tdnnf2 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf3 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf4 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=1
tdnnf-layer name=tdnnf5 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=0
tdnnf-layer name=tdnnf6 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf7 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf8 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf9 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf10 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf11 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf12 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
tdnnf-layer name=tdnnf13 l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66 dim=1024 bottleneck-dim=128 time-stride=3
linear-component name=prefinal-l dim=256 l2-regularize=0.008 orthonormal-constraint=-1.0

prefinal-layer name=prefinal-0-chain input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
output-layer name=output-0 include-log-softmax=false dim=3456 l2-regularize=0.002
prefinal-layer name=prefinal-1-chain input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
output-layer name=output-1 include-log-softmax=false dim=3456 l2-regularize=0.002

prefinal-layer name=prefinal-0-xent input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
output-layer name=output-0-xent dim=3456 learning-rate-factor=5.0 l2-regularize=0.002
prefinal-layer name=prefinal-1-xent input=prefinal-l l2-regularize=0.008 big-dim=1024 small-dim=256
output-layer name=output-1-xent dim=3456 learning-rate-factor=5.0 l2-regularize=0.002

component name=md-chain type=FixedAffineComponent matrix=code-mix-file/M.mat
component-node name=md-chain component=md-chain input=output-0.affine
output-node name=output-0 input=md-chain objective=linear
component name=md-xent type=FixedAffineComponent matrix=code-mix-file/M.mat
component-node name=md-xent component=md-xent input=output-0-xent.log-softmax
output-node name=output-0-xent input=md-xent objective=linear

component name=en-chain type=FixedAffineComponent matrix=code-mix-file/E.mat
component-node name=en-chain component=en-chain input=output-1.affine
output-node name=output-1 input=en-chain objective=linear
component name=en-xent type=FixedAffineComponent matrix=code-mix-file/E.mat
component-node name=en-xent component=en-xent input=output-1-xent.log-softmax
output-node name=output-1-xent input=en-xent objective=linear

output-node name=output input=Sum(md-chain, en-chain) objective=linear
output-node name=output-xent input=Sum(md-xent, en-xent) objective=linear
'''


def constrain_orthonormal_hook(model, unused_x):
    if not model.training:
        return
    
    model.ortho_constrain_count = (model.ortho_constrain_count + 1) % 2
    if model.ortho_constrain_count != 0:
        return

    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, 'constrain_orthonormal'):
                m.constrain_orthonormal()


# Create a network like the above one
class ChainModel(nn.Module):

    def __init__(self,
                 feat_dim,
                 output_dim,
                 ivector_dim=0,
                 lda_mat_filename=None,
                 hidden_dim=1024,
                 bottleneck_dim=128,
                 prefinal_bottleneck_dim=256,
                 kernel_size_list=[3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3],
                 subsampling_factor_list=[1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1],
                 frame_subsampling_factor=3):
        super().__init__()

        # at present, we support only frame_subsampling_factor to be 3
        assert frame_subsampling_factor == 3

        assert len(kernel_size_list) == len(subsampling_factor_list)
        num_layers = len(kernel_size_list)
        
        self.ortho_constrain_count = 0

        input_dim = feat_dim * 3 + ivector_dim
        
        self.tdnn1 = TDNN(input_dim=input_dim, hidden_dim=hidden_dim)

        tdnnfs = []
        for i in range(num_layers):
            kernel_size = kernel_size_list[i]
            subsampling_factor = subsampling_factor_list[i]
            layer = FactorizedTDNN(dim=hidden_dim,
                                   bottleneck_dim=bottleneck_dim,
                                   kernel_size=kernel_size,
                                   subsampling_factor=subsampling_factor)
            tdnnfs.append(layer)

        # tdnnfs requires [N, C, T]
        self.tdnnfs = nn.ModuleList(tdnnfs)

        # prefinal_l affine requires [N, C, T]
        self.prefinal_l = OrthonormalLinear(
            dim=hidden_dim,
            bottleneck_dim=prefinal_bottleneck_dim,
            kernel_size=1)

        # prefinal_chain requires [N, C, T]
        self.prefinal_1_chain = PrefinalLayer(big_dim=hidden_dim,
                                            small_dim=prefinal_bottleneck_dim)

        # output_affine requires [N, T, C]
        self.output_1_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                       out_features=output_dim)

        # prefinal_xent requires [N, C, T]
        self.prefinal_1_xent = PrefinalLayer(big_dim=hidden_dim,
                                           small_dim=prefinal_bottleneck_dim)

        self.output_1_xent_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                            out_features=output_dim)

        # prefinal_chain requires [N, C, T]
        self.prefinal_2_chain = PrefinalLayer(big_dim=hidden_dim,
                                            small_dim=prefinal_bottleneck_dim)

        # output_affine requires [N, T, C]
        self.output_2_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                       out_features=output_dim)

        # prefinal_xent requires [N, C, T]
        self.prefinal_2_xent = PrefinalLayer(big_dim=hidden_dim,
                                           small_dim=prefinal_bottleneck_dim)

        self.output_2_xent_affine = nn.Linear(in_features=prefinal_bottleneck_dim,
                                            out_features=output_dim)

        if lda_mat_filename:
            logging.info('Use LDA from {}'.format(lda_mat_filename))
            self.lda_A, self.lda_b = load_lda_mat(lda_mat_filename)
            assert input_dim == self.lda_A.shape[0]
            self.has_LDA = True
        else:
            logging.info('replace LDA with BatchNorm')
            self.input_batch_norm = nn.BatchNorm1d(num_features=input_dim,
                                                   affine=False)
            self.has_LDA = False

        self.register_forward_pre_hook(constrain_orthonormal_hook)

    def forward(self, x, dropout=0. , lang=None, entropy=None):
        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        assert x.ndim == 3

        self.lang = lang

        if self.has_LDA:
            # to() does not copy data if lda_A is already in the expected device
            self.lda_A = self.lda_A.to(x.device)
            self.lda_b = self.lda_b.to(x.device)

            x = torch.matmul(x, self.lda_A) + self.lda_b

            # at this point, x is [N, T, C]

            x = x.permute(0, 2, 1)
        else:
            # at this point, x is [N, T, C]
            x = x.permute(0, 2, 1)
            # at this point, x is [N, C, T]
            x = self.input_batch_norm(x)

        # at this point, x is [N, C, T]

        x = self.tdnn1(x, dropout=dropout)

        # tdnnf requires input of shape [N, C, T]
        for i in range(len(self.tdnnfs)):
            x = self.tdnnfs[i](x, dropout=dropout)

        # at this point, x is [N, C, T]

        x = self.prefinal_l(x)

        # at this point, x is [N, C, T]

        if self.lang == int(1):
            # for the output node '1'
            print("use output 1")
            nnet_output_1 = self.prefinal_1_chain(x)

            # at this point, nnet_output is [N, C, T]
            nnet_output_1 = nnet_output_1.permute(0, 2, 1)
            # at this point, nnet_output is [N, T, C]
            nnet_output_1 = self.output_1_affine(nnet_output_1)

            # for the xent node
            xent_output_1 = self.prefinal_1_xent(x)

            # at this point, xent_output is [N, C, T]
            xent_output_1 = xent_output_1.permute(0, 2, 1)
            # at this point, xent_output is [N, T, C]
            xent_output_1 = self.output_1_xent_affine(xent_output_1)

            xent_output_1 = F.log_softmax(xent_output_1, dim=-1)

            with torch.no_grad():
                nnet_output_2 = self.prefinal_2_chain(x)
                nnet_output_2 = nnet_output_2.permute(0, 2, 1)
                nnet_output_2 = self.output_2_affine(nnet_output_2)
                xent_output_2 = self.prefinal_2_xent(x)
                xent_output_2 = xent_output_2.permute(0, 2, 1)
                xent_output_2 = self.output_2_xent_affine(xent_output_2)
                xent_output_2 = F.log_softmax(xent_output_2, dim=-1)
                nnet_output = (nnet_output_1 + nnet_output_2) / 2
                xent_output = (xent_output_1 + xent_output_2) / 2

            return nnet_output_1, xent_output_1

        elif self.lang == int(2):
            # for the output node '2'
            print("use output 2")
            nnet_output_2 = self.prefinal_2_chain(x)

            # at this point, nnet_output is [N, C, T]
            nnet_output_2 = nnet_output_2.permute(0, 2, 1)
            # at this point, nnet_output is [N, T, C]
            nnet_output_2 = self.output_2_affine(nnet_output_2)

            # for the xent node
            xent_output_2 = self.prefinal_2_xent(x)

            # at this point, xent_output is [N, C, T]
            xent_output_2 = xent_output_2.permute(0, 2, 1)
            # at this point, xent_output is [N, T, C]
            xent_output_2 = self.output_2_xent_affine(xent_output_2)

            xent_output_2 = F.log_softmax(xent_output_2, dim=-1)
            
            with torch.no_grad():
                nnet_output_1 = self.prefinal_1_chain(x)
                nnet_output_1 = nnet_output_1.permute(0, 2, 1)
                nnet_output_1 = self.output_1_affine(nnet_output_1)
                xent_output_1 = self.prefinal_1_xent(x)
                xent_output_1 = xent_output_1.permute(0, 2, 1)
                xent_output_1 = self.output_1_xent_affine(xent_output_1)
                xent_output_1 = F.log_softmax(xent_output_1, dim=-1)
                nnet_output = (nnet_output_1 + nnet_output_2) / 2
                xent_output = (xent_output_1 + xent_output_2) / 2

            return nnet_output_2, xent_output_2

        else:
            # for inference use
            #TODO
            with torch.no_grad():
                # 1. pdf select combined
                nnet_output_1 = self.prefinal_1_chain(x)
                nnet_output_1 = nnet_output_1.permute(0, 2, 1)
                nnet_output_1 = self.output_1_affine(nnet_output_1)
                xent_output_1 = self.prefinal_1_xent(x)
                xent_output_1 = xent_output_1.permute(0, 2, 1)
                xent_output_1 = self.output_1_xent_affine(xent_output_1)
                xent_output_1 = F.log_softmax(xent_output_1, dim=-1)

                nnet_output_2 = self.prefinal_2_chain(x)
                nnet_output_2 = nnet_output_2.permute(0, 2, 1)
                nnet_output_2 = self.output_2_affine(nnet_output_2)
                xent_output_2 = self.prefinal_2_xent(x)
                xent_output_2 = xent_output_2.permute(0, 2, 1)
                xent_output_2 = self.output_2_xent_affine(xent_output_2)
                xent_output_2 = F.log_softmax(xent_output_2, dim=-1)

                nnet_output = pdf_select(nnet_output_1, nnet_output_2)
                xent_output = pdf_select(xent_output_1, xent_output_2)

                if entropy:
                    nnet_output = EntropyFunction.apply(nnet_output)
                    xent_output = EntropyFunction.apply(xent_output)

                return nnet_output, xent_output


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    feat_dim = 40
    output_dim = 288
    model = ChainModel(feat_dim=feat_dim, output_dim=output_dim)
    #  logging.info(model)
    N = 1
    T = 150 + 27 + 27
    C = feat_dim * 3
    x = torch.arange(N * T * C).reshape(N, T, C).float()
    nnet_output, xent_output = model(x, lang=None)
    print(x.shape, nnet_output.shape, xent_output.shape)
    for name, p in model.named_parameters():
            print(name, p)