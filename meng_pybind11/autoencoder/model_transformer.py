import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_transformer_model(ninp):
    model = TransformerModel(ninp)

    return model

class TransformerModel(nn.Module):

    def __init__(self,
                 ninp=40, # encoder/decoder input dim == C
                 nhead=8,
                 nhid=1024, # feedforward dim
                 en_nlayers=6,
                 de_nlayers=6,
                 dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, en_nlayers) # shape [N, T, C]
#        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
#        decoder_layers = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
#        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, de_nlayers)
        self.decoder = nn.Linear(ninp, ninp)

#        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

#    def forward(self, src):
#        if self.src_mask is None or self.src_mask.size(0) != len(src):
#            device = src.device
#            mask = self._generate_square_subsequent_mask(len(src)).to(device)
#            self.src_mask = mask
#
#        src = self.encoder(src) * math.sqrt(self.ninp)
#        src = self.pos_encoder(src)
#        output = self.transformer_encoder(src, self.src_mask)
#        output = self.decoder(output)
#        return output

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

        # input x is of shape: [batch_size, seq_len, feat_dim] = [N, T, C]
        x = feat
     
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x)
        x = self.decoder(x)
#        x = x.permute(0, 2, 1)

        return x, feat_len_list

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def _test_transformer_model():
    model = TransformerModel(
                     ninp=40)

    feat1 = torch.randn((6, 40))
    feat2 = torch.randn((8, 40))

    from torch.nn.utils.rnn import pad_sequence
    feat = pad_sequence([feat1, feat2], batch_first=True)

    feat_len_list = [6, 8]

    x, _ = model(feat, feat_len_list)

    assert x.shape == torch.Size([2, 8, 40])

if __name__ == '__main__':
    _test_transformer_model()
