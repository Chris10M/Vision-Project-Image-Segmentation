from logger import setup_logger
import torch.multiprocessing as mp
from model import model
import os
from cityscapes import CityScapes
from loss import OhemCELoss, IoULoss, OHIoULoss, DiceLoss
from evaluate import evaluate_net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import hashlib
import os
import os.path as osp
import logging
import time
import datetime
import imutils

from arg_parser import train


class Logger:
    logger = None
    ModelSavePath = 'model'

def set_model_logger(net):
    model_info = str(net)

    respth = f'savedmodels/{hashlib.md5(model_info.encode()).hexdigest()}'
    Logger.ModelSavePath = respth

    if not osp.exists(respth): os.makedirs(respth)
    logger = logging.getLogger()

    if setup_logger(respth):
        logger.info(model_info)

    Logger.logger = logger


def main(args):
    scale = 0.5
    cropsize = [int(2048 * scale), int(1024 * scale)]
    ds = CityScapes(args.cityscapes_path, cropsize=cropsize, mode='train')

    n_classes = ds.n_classes    
    net = model.get_network(n_classes)

    set_model_logger(net)

    saved_path = args.saved_model
    
    max_iter = 6400000
    optim_iter = 10
    save_iter = 1000
    n_img_per_gpu = 2
    n_workers = 8
    
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=True,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    ## model
    ignore_idx = 255

    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    optim = torch.optim.Adam(net.parameters())

    epoch = 0
    start_it = 0
    if os.path.isfile(saved_path):
        loaded_model = torch.load(saved_path)
        state_dict = loaded_model['state_dict']

        net.load_state_dict(state_dict, strict=False)

        try:
            optim.load_state_dict(loaded_model['optimize_state'])
            ...
        except ValueError: pass

        try:
            start_it = loaded_model['start_it'] + 2
        except KeyError:
            start_it = 0

        try:
            epoch = loaded_model['epoch']
        except KeyError:
            epoch = 0

        print(f'Model Loaded: {saved_path} @ start_it: {start_it}')

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)

    start_training = False
    for it in range(start_it, max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb = next(diter)

        im = im.cuda()
        lb = lb.cuda()

        if not start_training:
            start_training = True

        outs = net(im)
        if isinstance(outs, tuple):  # Depending on the model, AuxLoss may be also computed.
            out, aux_loss_1, aux_loss_2 = outs
            loss = criteria(out, lb) + 0.4 * criteria(aux_loss_1, lb) + 0.6 * criteria(aux_loss_2, lb)
        else:
            out = outs
            loss = criteria(out, lb)

        loss /= optim_iter

        loss.backward()

        if it % optim_iter == 0:  # we optimize the loss only for every optim_iter. This done to mimic the increase in batch size.
            optim.step()
            optim.zero_grad()

        loss_avg.append(loss.item())

        if (it + 1) % save_iter == 0 or os.path.isfile('save'):
            save_pth = osp.join(Logger.ModelSavePath, f'{it + 1}_{int(time.time())}.pth')

            Logger.logger.info(f"Model@{it + 1}\n{evaluate_net(args, net)}")

            print(f'Saving model at: {(it + 1)}')
            torch.save({
                'epoch': epoch,
                'start_it': it,
                'state_dict': net.state_dict(),
                'optimize_state': optim.state_dict()
            }, save_pth)
            print(f'model at: {(it + 1)} Saved')
            ds.shuffle()

        #   print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    f'epoch: {epoch}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            Logger.logger.info(msg)
            loss_avg = []
            st = ed

    save_pth = osp.join(Logger.ModelSavePath, 'model_final.pth')
    net.cpu()
    torch.save({'state_dict': net.state_dict()}, save_pth)
            
    Logger.logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    args = train()
    main(args)
