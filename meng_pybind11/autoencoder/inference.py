import logging
import os
import sys

import torch
import torch.nn.functional as F

import kaldi

from common import load_checkpoint
from common import setup_logger
from dataset import get_aed_dataloader
from model import get_aed_model
from options import get_args


def main():
    args = get_args()

    setup_logger('{}/log-inference'.format(args.dir), args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.warning('No GPU detected! Use CPU for inference.')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device_id)

    model = get_aed_model(
        input_dim=args.input_dim,
        output_dim=args.output_dim)

    load_checkpoint(args.checkpoint, model)

    model.to(device)
    model.eval()

    wspecifier = 'ark,scp:{filename}.ark,{filename}.scp'.format(
        filename=os.path.join(args.dir, 'nnet_output'))

    writer = kaldi.MatrixWriter(wspecifier)

    dataloader = get_aed_dataloader(
        feats_scp=args.feats_scp,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8)


    for batch_idx, batch in enumerate(dataloader):
        uttid_list, feat, feat_len_list, _, _ = batch

        feat = feat.to(device)

        with torch.no_grad():
            activations, feat_len_list = model(feat, feat_len_list)

        num = len(uttid_list)
        for i in range(num):
            uttid = uttid_list[i]
            feat_len = feat_len_list[i]
            value = activations[i, :feat_len, :]

            value = value.cpu()

            writer.Write(uttid, value.numpy())

        if batch_idx % 10 == 0:
            logging.info('Processed batch {}/{} ({:.3f}%)'.format(
                batch_idx, len(dataloader),
                float(batch_idx) / len(dataloader) * 100))

    writer.Close()
    logging.info('pseudo-log-likelihood is saved to {}'.format(
        os.path.join(args.dir, 'nnet_output.scp')))


if __name__ == '__main__':
    main()











