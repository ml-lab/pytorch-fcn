#!/usr/bin/env python

import argparse
import os.path as osp

import fcn
import pandas
import torch
from torch.autograd import Variable
import torchfcn
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('model_path')
args = parser.parse_args()

n_class = 21
model = torchfcn.models.FCN32s(n_class=n_class, deconv=True)
model.load_state_dict(torch.load(args.model_path))
model = model.cuda()

root = osp.expanduser('~/data/datasets')
loader = torch.utils.data.DataLoader(
    torchfcn.datasets.VOC2011ClassSeg(
        root, split='seg11valid', transform=True),
    batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
metrics = []
for batch_idx, (data, target) in tqdm.tqdm(
        enumerate(loader), total=len(loader), ncols=80):
    data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    score = model(data)

    lp = score.data.max(1)[1].cpu().numpy()[:, 0, :, :][0]
    lt = target.data.cpu().numpy()[0]
    acc, acc_cls, mean_iu, fwavacc = fcn.utils.label_accuracy_score(
        lt, lp, n_class=n_class)
    metrics.append({
        'acc': acc,
        'acc_cls': acc_cls,
        'mean_iu': mean_iu,
        'fwavacc': fwavacc,
    })
print(pandas.DataFrame(metrics).mean(axis=0))
