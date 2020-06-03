import logging
import math
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

import kaldi
import torch
import torch.nn.functional as F
import torch.nn as nn

from common import load_checkpoint
from common import save_checkpoint
from common import save_training_info
from common import setup_logger
from dataset import get_aed_dataloader
from options import get_args
from model import get_aed_model

def train_one_epoch(dataloader, model, device, optimizer,
                     current_epoch, tf_writer):

    total_loss = 0.
    num = 0.

    criterion = nn.L1Loss()

    num_repeat = 1
    for kk in range(num_repeat):
        for batch_idx, batch in enumerate(dataloader):
            uttid_list, feat, feat_len_list, target, target_len_list = batch

            feat = feat.to(device)
            target = target.to(device)

            activations, feat_len_list = model(feat, feat_len_list)

            loss = criterion(activations, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            num += 1

            if batch_idx % 100 == 0:
                logging.info(
                    'batch {}/{} ({:.2f}%) ({}/{}), loss {:.5f}, average {:.5f}'
                    .format(batch_idx, len(dataloader),
                            float(batch_idx) / len(dataloader) * 100, kk,
                            num_repeat, loss.item(), total_loss / num))

            if batch_idx % 100 == 0:
                tf_writer.add_scalar(
                    'train/current_batch_average_loss', loss.item(),
                    batch_idx + kk * len(dataloader) +
                    num_repeat * len(dataloader) * current_epoch)

                tf_writer.add_scalar(
                    'train/global_average_loss', total_loss / num,
                    batch_idx + kk * len(dataloader) +
                    num_repeat * len(dataloader) * current_epoch)

    return total_loss / num


def main():
    args = get_args()
    setup_logger('{}/log-train'.format(args.dir), args.log_level)
    logging.info(' '.join(sys.argv))

    if torch.cuda.is_available() == False:
        logging.error('No GPU detected!')
        sys.exit(-1)

    kaldi.SelectGpuDevice(device_id=args.device_id)

    device = torch.device('cuda', args.device_id)

    model = get_aed_model(
        input_dim=args.input_dim,
        output_dim=args.output_dim)

    start_epoch = 0
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    best_loss = None

    if args.checkpoint:
        start_epoch, learning_rate, best_loss = load_checkpoint(
            args.checkpoint, model)
        logging.info(
            'loaded from checkpoint: start epoch {start_epoch}, '
            'learning rate {learning_rate}, best loss {best_loss}'.format(
                start_epoch=start_epoch,
                learning_rate=learning_rate,
                best_loss=best_loss))

    model.to(device)

    dataloader = get_aed_dataloader(
        feats_scp=args.feats_scp,
        targets_scp=args.targets_scp,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)

    lr = learning_rate
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=args.l2_regularize)

    tf_writer = SummaryWriter(log_dir='{}/tensorboard'.format(args.dir))

    model.train()

    best_epoch = 0
    best_model_path = os.path.join(args.dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(args.dir, 'best-epoch-info')

    try:
        for epoch in range(start_epoch, num_epochs):
            learning_rate = lr * pow(0.8, epoch)
            tf_writer.add_scalar('learning_rate', learning_rate, epoch)

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            logging.info('epoch {}, learning rate {}'.format(
                epoch, learning_rate))

            loss = train_one_epoch(dataloader=dataloader,
                                   model=model,
                                   device=device,
                                   optimizer=optimizer,
                                   current_epoch=epoch,
                                   tf_writer=tf_writer)

             # the loss, the better
            if best_loss is None or best_loss > loss:
                best_loss = loss
                best_epoch = epoch
                save_checkpoint(filename=best_model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                loss=loss)
                save_training_info(filename=best_epoch_info_filename,
                                   model_path=best_model_path,
                                   current_epoch=epoch,
                                   learning_rate=learning_rate,
                                   loss=loss,
                                   best_loss=best_loss,
                                   best_epoch=best_epoch)

            # we always save the model for every epoch
            model_path = os.path.join(args.dir, 'epoch-{}.pt'.format(epoch))
            save_checkpoint(filename=model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=learning_rate,
                            loss=loss)

            epoch_info_filename = os.path.join(args.dir,
                                               'epoch-{}-info'.format(epoch))
            save_training_info(filename=epoch_info_filename,
                               model_path=model_path,
                               current_epoch=epoch,
                               learning_rate=learning_rate,
                               loss=loss,
                               best_loss=best_loss,
                               best_epoch=best_epoch)


    except KeyboardInterrupt:
        # save the model when ctrl-c is pressed
        model_path = os.path.join(args.dir,
                                  'epoch-{}-interrupted.pt'.format(epoch))
        # use a very large loss for the interrupted model
        loss = 100000000
        save_checkpoint(model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=learning_rate,
                        loss=loss)

        epoch_info_filename = os.path.join(
            args.dir, 'epoch-{}-interrupted-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=learning_rate,
                           loss=loss,
                           best_loss=best_loss,
                           best_epoch=best_epoch)

    tf_writer.close()
    logging.warning('Training done!')


if __name__ == '__main__':
    np.random.seed(20200302)
    torch.manual_seed(20200302)
    main()
