#!/usr/bin/env python3

# Copyright 2020 Author: Meng Wu
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class LanguageEntropyControl(nn.Module):
    def __init__(self, normalize=None, cost='default'):
        '''
        '''
        super().__init__()
        self.entropy_normalize = normalize
        self.cost_type = cost
        self.sil_pdf = [0, 1, 2, 76, 229]
        self.en_pdf = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                    33, 34, 35, 36, 37, 38, 39, 40, 41, 82, 83, 86, 88, 90, 91,
                    92, 96, 99, 100, 103, 104, 105, 106, 108, 109, 110, 111, 112,
                    120, 121, 123, 124, 125, 126, 127, 129, 130, 131, 137, 138, 139,
                    151, 154, 159, 160, 170, 173, 178, 180, 182, 185, 189, 192, 199,
                    211, 214, 221, 227, 230, 235, 238, 239, 241, 242, 244, 247, 248,
                    249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262,
                    263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 279, 280, 281, 283, 284, 285, 286, 287]
        self.md_pdf = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                    72, 73, 74, 75, 77, 78, 79, 80, 81, 84, 85, 87, 89, 93, 94, 95,
                    97, 98, 101, 102, 107, 113, 114, 115, 116, 117, 118, 119, 122, 128,
                    132, 133, 134, 135, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148,
                    149, 150, 152, 153, 155, 156, 157, 158, 161, 162, 163, 164, 165, 166,
                    167, 168, 169, 171, 172, 174, 175, 176, 177, 179, 181, 183, 184, 186,
                    187, 188, 190, 191, 193, 194, 195, 196, 197, 198, 200, 201, 202, 203,
                    204, 205, 206, 207, 208, 209, 210, 212, 213, 215, 216, 217, 218, 219,
                    220, 222, 223, 224, 225, 226, 228, 231, 232, 233, 234, 236, 237, 240, 243, 245, 246, 258, 275, 277, 278, 282]
        self.num_pdf = len(self.sil_pdf + self.en_pdf + self.md_pdf)
        assert self.num_pdf == 288
        self.sil_pdf_vec = torch.Tensor([ 1 if i in self.sil_pdf else 0 for i in range(self.num_pdf) ])
        self.sil_pdf_vec = self.sil_pdf_vec.view(1, 1, -1)
        self.en_pdf_vec = torch.Tensor([ 1 if i in self.en_pdf else 0 for i in range(self.num_pdf) ])
        self.en_pdf_vec = self.en_pdf_vec.view(1, 1, -1)
        self.md_pdf_vec = torch.Tensor([ 1 if i in self.md_pdf else 0 for i in range(self.num_pdf) ])
        self.md_pdf_vec = self.md_pdf_vec.view(1, 1, -1)
        
    
    def forward(self, x):
        N, T, _ = x.shape
        # x is [N, T, C]

        prob = F.softmax(x, dim=-1).detach()

        sil_pdf_vec = self.sil_pdf_vec.to(x.device)
        en_pdf_vec = self.en_pdf_vec.to(x.device)
        md_pdf_vec = self.md_pdf_vec.to(x.device)

        sil_prob = prob * sil_pdf_vec

        en_prob = prob * en_pdf_vec # p select
        en_plogp = en_prob * torch.log(en_prob) # compute Plog(P)
        en_plogp[ en_plogp != en_plogp ] = 0 # convert nan to 0
        # now en_plog is [N, T, C]
        en_entropy = (-1) * en_plogp.view(N, T, -1).sum(2) # [N,T] for each point in T means each frames En entropy
        en_entropy = en_entropy.view(N, T, 1) # [N, T, 1]

        md_prob = prob * md_pdf_vec
        md_plogp = md_prob * torch.log(md_prob) # compute Plog(P)
        md_plogp[ md_plogp != md_plogp ] = 0 # convert nan to 0
        # now md_plog is [N, T, C]
        md_entropy = (-1) * md_plogp.view(N, T, -1).sum(2) # [N,T] for each point in T means each frames Md entropy
        md_entropy = md_entropy.view(N, T, 1) # [N, T, 1]

        if self.entropy_normalize:
            # normalize by each class length
            en_entropy = en_entropy / torch.log(torch.Tensor([len(self.en_pdf)]).to(x.device))
            md_entropy = md_entropy / torch.log(torch.Tensor([len(self.md_pdf)]).to(x.device))

        if self.cost_type == 'alex':
            md_cost = torch.sqrt(en_entropy * md_entropy) / md_entropy
            en_cost = torch.sqrt(en_entropy * md_entropy) / en_entropy
            x = torch.log( sil_prob + en_prob * en_cost + md_prob * md_cost)
            return x
        else:
            ratio = md_entropy > en_entropy
            md_cost = en_entropy / md_entropy
            en_cost = md_entropy / en_entropy
            for i in range(N):
                for j in range(T):
                    if ratio[i][j]:
                        prob[i][j] = md_cost[i][j] * md_prob[i][j] + en_prob[i][j] + sil_prob[i][j]
                    else:
                        prob[i][j] = en_cost[i][j] * en_prob[i][j] + md_prob[i][j] + sil_prob[i][j]
            return torch.log(prob)


def pdf_select(x1, x2):
    #print('x is', x)
    sil_pdf = [0, 1, 2, 76, 229]
    en_pdf = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 82, 83, 86, 88, 90, 91,
        92, 96, 99, 100, 103, 104, 105, 106, 108, 109, 110, 111, 112,
        120, 121, 123, 124, 125, 126, 127, 129, 130, 131, 137, 138, 139,
        151, 154, 159, 160, 170, 173, 178, 180, 182, 185, 189, 192, 199,
        211, 214, 221, 227, 230, 235, 238, 239, 241, 242, 244, 247, 248,
        249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262,
        263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 279, 280, 281, 283, 284, 285, 286, 287]
    md_pdf = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
        72, 73, 74, 75, 77, 78, 79, 80, 81, 84, 85, 87, 89, 93, 94, 95,
        97, 98, 101, 102, 107, 113, 114, 115, 116, 117, 118, 119, 122, 128,
        132, 133, 134, 135, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148,
        149, 150, 152, 153, 155, 156, 157, 158, 161, 162, 163, 164, 165, 166,
        167, 168, 169, 171, 172, 174, 175, 176, 177, 179, 181, 183, 184, 186,
        187, 188, 190, 191, 193, 194, 195, 196, 197, 198, 200, 201, 202, 203,
        204, 205, 206, 207, 208, 209, 210, 212, 213, 215, 216, 217, 218, 219,
        220, 222, 223, 224, 225, 226, 228, 231, 232, 233, 234, 236, 237, 240, 243, 245, 246, 258, 275, 277, 278, 282]
    num_pdf = len(sil_pdf + en_pdf + md_pdf)
    assert num_pdf == 288
    sil_pdf_vec = torch.Tensor([ 1 if i in sil_pdf else 0 for i in range(num_pdf) ])
    sil_pdf_vec = sil_pdf_vec.view(1, 1, -1)
    en_pdf_vec = torch.Tensor([ 1 if i in en_pdf else 0 for i in range(num_pdf) ])
    en_pdf_vec = en_pdf_vec.view(1, 1, -1)
    md_pdf_vec = torch.Tensor([ 1 if i in md_pdf else 0 for i in range(num_pdf) ])
    md_pdf_vec = md_pdf_vec.view(1, 1, -1)

    # x is [N, T, C]
    prob_1 = F.softmax(x1.detach(), dim=-1)
    prob_2 = F.softmax(x2.detach(), dim=-1)
    
    sil_pdf_vec = sil_pdf_vec.to(x1.device).detach()
    en_pdf_vec = en_pdf_vec.to(x1.device).detach()
    md_pdf_vec = md_pdf_vec.to(x1.device).detach()

    sil_prob = (prob_1 * sil_pdf_vec + prob_2 * sil_pdf_vec)/2
    en_prob = prob_2 * en_pdf_vec
    md_prob = prob_1 * md_pdf_vec
    
    x = torch.log( sil_prob + en_prob + md_prob)
    
    return x


def _test_entropy_layer():
    N = 3
    T = 20
    C = 288
    x = torch.rand(N, T, C, dtype=torch.float)
    entropy_layer = LanguageEntropyControl(normalize=True, cost='alex')
    y = entropy_layer(x)
    print(x)
    print(y)

def _test_pdf_select():
    N = 3
    T = 20
    C = 288
    x1 = torch.rand(N, T, C, dtype=torch.float)
    x2 = torch.rand(N, T, C, dtype=torch.float)
    y = pdf_select(x1, x2)
    print(y.shape)

if __name__ == '__main__':
    #_test_entropy_layer()
    _test_pdf_select()