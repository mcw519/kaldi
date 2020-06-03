import kaldi
import torch

import os
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_aed_dataloader(feats_scp,
                       targets_scp=None,
                       batch_size=1,
                       shuffle=False,
                       num_workers=0):

    dataset = AedDataset(feats_scp=feats_scp, targets_scp=targets_scp)

    collate_fn = AedDatasetCollateFunc()

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    return dataloader


class AedDataset(Dataset):

    def __init__(self, feats_scp, targets_scp=None):
        assert os.path.isfile(feats_scp)
        if targets_scp:
            assert os.path.isfile(targets_scp)
            logging.info('targets scp: {}'.format(targets_scp))
        else:
            logging.warn('No targets scp: {}'.format(targets_scp))

        items = dict()
        with open(feats_scp, 'r') as f:
            for line in f:
                # every line has the following format:
                # uttid rxfilename
                uttid_rxfilename = line.split()
                assert len(uttid_rxfilename) == 2

                uttid, rxfilename = uttid_rxfilename

                assert uttid not in items

                items[uttid] = [uttid, rxfilename, None]

        expected_count = len(items)
        n = 0

        if targets_scp:
            expected_count = len(items)
            n = 0
            with open(targets_scp, 'r') as f:
                for line in f:
                    uttid_rxfilename = line.split()
                    assert len(uttid_rxfilename) == 2

                    uttid, rxfilename = uttid_rxfilename

                    assert uttid in items

                    items[uttid][-1] = rxfilename

                    n = n +1

            # every utterance should have a target if targets_scp is given
            assert n == expected_count

        self.items = list(items.values())
        self.num_items = len(self.items)
        self.feats_scp = feats_scp
        self.targets_scp = targets_scp

    def __len__(self):
        return self.num_items

    def __getitem__(self, i):
        '''
        Returns:
            a list [key, feat_rxfilename, target_rxfilename]
        '''
        return self.items[i]

    def __str__(self):
        s = 'feats scp: {}\n'.format(self.feats_scp)

        s += 'target scp: {}\n'.format(self.targets_scp)

        s += 'num utterances: {}\n'.format(self.num_items)

        return s


class AedDatasetCollateFunc:

    def __call__(self, batch):
        '''
        Args:
            batch: a list of [uttid, feat_rxfilename, target_rxfilename].

        Returns:
            uttid_list: a list ot utterance id

            feat: a 3-D float tensor of shape [batch_size, seq_len, feat_dim]

            feat_len_list: number of frames of each utterance before padding

            target_list: a list of target of each utterance

            target_len_list: target length of each utterance
        '''
        uttid_list = [] # utterance id of each utterance
        feat_len_list = [] # number of frames of each utterance
        target_list = [] # target of each utterance
        target_len_list = [] # target length of each utterance

        feat_list = []

        for b in batch:
            uttid, feat_rxfilename, target_rxfilename = b

            uttid_list.append(uttid)
            feat = kaldi.read_mat(feat_rxfilename).numpy()
            feat = torch.from_numpy(feat).float()
            feat_list.append(feat)
            feat_len_list.append(feat.size(0))

            if target_rxfilename:
                target = kaldi.read_mat(target_rxfilename).numpy()
                target = torch.from_numpy(target).float()
                target_list.append(target)
                target_len_list.append(target.size(0))

        feat = pad_sequence(feat_list, batch_first=True)

        if not target_list:
            target = None
            target_len_list = None
        else:
            target = pad_sequence(target_list, batch_first=True)

        return uttid_list, feat, feat_len_list, target, target_len_list


def _test_dataset():
    feats_scp = 'autoencoder/feats.scp'
    targets_scp = 'autoencoder/targets.scp'

    dataset = AedDataset(feats_scp=feats_scp, targets_scp=targets_scp)

    print(dataset)

def _test_dataloader():
    feats_scp = 'autoencoder/feats.scp'
    targets_scp = 'autoencoder/targets.scp'

    dataset = AedDataset(feats_scp=feats_scp, targets_scp=targets_scp)

    dataloader = DataLoader(dataset,
                            batch_size=2,
                            num_workers=5,
                            shuffle=True,
                            collate_fn=AedDatasetCollateFunc())

    i = 0
    for batch in dataloader:
        uttid_list, feat, feat_len_list, target, target_len_list = batch
        print(uttid_list, feat.shape, feat_len_list, target.shape, target_len_list)
        i += 1
        if i > 10:
            break

if __name__ == '__main__':
    _test_dataloader()

