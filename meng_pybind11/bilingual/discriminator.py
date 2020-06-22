#!/usr/bin/env python3

# Copyright 2020 Author: Meng Wu

import torch
import torch.nn as nn
import torch.nn.functional as F
import openfst_python as pyfst


class LIDDiscriminator(nn.Module):
    def __init__(self, input_dim=256 , hidden_dim=128, output_dim=3):
        super().__init__()
        self.lid_pre1 = nn.Linear(input_dim, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_dim, affine=False)
        self.lid_pre2 = nn.Linear(hidden_dim, hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_dim, affine=False)
        self.lid_pred = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x is [N, C, T]
        x = x.permute(0, 2, 1)
        # at this point, x is [N, T, C]
        x = self.lid_pre1(x)
        x = F.relu(x)
        # at this point, x is [N, T, C]
        x = x.permute(0, 2, 1)
        # at this point, x is [N, C, T]
        x = self.batchnorm1(x)
        x = x.permute(0, 2, 1)
        # at this point, x is [N, T, C]
        x = self.lid_pre2(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        # at this point, x is [N, C, T]
        x = self.batchnorm2(x)
        x = x.permute(0, 2, 1)
        # at this point, x is [N, T, C]
        x = self.lid_pred(x)
        x = F.relu(x)

        return x


def supervision2langid(supervision):
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

    frames_per_sequence = supervision.frames_per_sequence
    num_sequences = supervision.num_sequences
    fst = []
    
    for i in supervision.fst.ToString().split('\n'):
        fst.append(i.strip().split())

    # structure
    # next_state, in_symbol, out_symbol, [weight]
    # 35:{'38': ['2', '2', 4.70703], '39': ['12', '12', 0.0], '37': ['258', '258', 0.0]}
    fst_dict = {}
    for i in range(len(fst) - 1):
        if i != (len(fst) - 2):
            try:
                _start, _next, _iid, _oid, _weight = fst[i]
            except ValueError:
                _start, _next, _iid, _oid = fst[i]
                _weight = 0.
            
            _start, _next, _iid, _oid = int(_start), int(_next), int(_iid), int(_oid)

            try:
                fst_dict[_start].update({_next : [_iid, _oid, float(_weight)]})
            except KeyError:
                fst_dict[_start] = {_next : [_iid, _oid, float(_weight)]}
        else:
            _end = fst[i][0]
            fst_dict[_end] = None
    
    f = pyfst.Fst()
    state_list = [ None for i in range(len(fst_dict)) ]
    for i in range(len(fst_dict)):
        state_list[i] = f.add_state()
    
    for i in range(len(fst_dict)-1):
        for j in fst_dict[i]:
            _iid, _oid, _weight = fst_dict[i][j]
            f.add_arc(state_list[i], pyfst.Arc(_iid, _oid, pyfst.Weight(f.weight_type(), _weight), state_list[j]))
    
    f.set_start(state_list[0])
    f.set_final(state_list[-1])
    short_path = pyfst.shortestpath(f)

    # short_path as Reverse states
    # convert to list format
    fst_list = []
    temp = short_path.text().split('\n')
    for i in temp:
        fst_list.append([ float(j) for j in i.strip().split() ])
    
    fst_list = sorted(fst_list, reverse=True)
    fst_list = list(filter(None, fst_list))
    # get target id
    target = []
    for i in range(len(fst_list) - 1):
        # fst symbol used pdf-id + 1, so minus one
        if fst_list[i][3] - 1 in sil_pdf:
            target.append(0)
        elif fst_list[i][3] - 1 in md_pdf:
            target.append(1)
        elif fst_list[i][3] - 1 in en_pdf:
            target.append(2)

    target = torch.Tensor(target).long()
    assert len(target) == frames_per_sequence * num_sequences

    return target


def lid_decode(nnet_output_1, nnet_output_2, lid):
    # lid is [N, T, 3], [0:sil, 1:md, 2:en]
    # nnet_output is [N, T, C]
    nnet_output_1 = F.softmax(nnet_output_1, dim=-1)
    nnet_output_2 = F.softmax(nnet_output_2, dim=-1)
    lang_id = F.softmax(lid, dim=-1)
    nnet_output = []
    xent_output = []
    for batch_idx, batch in enumerate(lang_id):
        temp = []
        xent_temp = []
        for seq_idx, value in enumerate(batch):
            sil, md, en = value
            _, idx = value.max(0)
            #print(idx)
            if idx == 0:
                temp.append((nnet_output_1[batch_idx][seq_idx]+nnet_output_2[batch_idx][seq_idx])/2)
            elif idx == 1:
                temp.append(md * nnet_output_1[batch_idx][seq_idx] + en * nnet_output_2[batch_idx][seq_idx])
            elif idx == 2:
                temp.append(md * nnet_output_1[batch_idx][seq_idx] + en * nnet_output_2[batch_idx][seq_idx])
                
        nnet_output.append(torch.stack(temp))
                
    nnet_output = torch.stack(nnet_output)

    return torch.log(nnet_output)


def _test():
    liddiscriminator = LIDDiscriminator(256, 128, 3)
    N = 3
    T = 10
    C = 256
    x = torch.rand(N*T*C).reshape(N, C, T)
    y = liddiscriminator(x)

    print(y)

if __name__ == '__main__':
    _test()