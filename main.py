import argparse, os, sys, subprocess
from tqdm import tqdm
from glob import glob
from os.path import *
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pickle as pkl

from models import VGG_graph_matching
from dataloader import MpiSintelClean, MpiSintelFinal, ImagesFromFolder
from logger import Logger
from utils import flow_utils, tools


def init_config():
    parser = argparse.ArgumentParser(description='Deep Learning of Graph matching')

    # dataset
    parser.add_argument('--dataset', type=str, default='sintel', help='dataset to use: middlebury/sintel')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='Perform Inference')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_flow', action='store_true', default=False, help='Save flow files during evaluation')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to dataset root directory')

    # others
    parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--use_vgg', action='store_true', default=True, help='Use VGG weights')



    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "models/%s" % args.dataset
    log_dir = "logs/%s" % args.dataset

    config_file = "config_%s" % args.dataset
    params = importlib.import_module(config_file).params

    args = argparse.Namespace(**vars(args), **params)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    id_ = "%s_seed-%d" % \
            (args.dataset, args.seed)

    save_path = os.path.join(save_dir, id_ + '.pt')

    args.save_path = save_path

    args.log_path = os.path.join(log_dir, id_ + ".log")
    print("log path", args.log_path)



    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True


    args.effective_number_workers = args.number_workers * args.number_gpus
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    return args

def _apply_loss(d, d_gt):

    # Set all pixel entries to 0 whose displacement magnitude is bigger than 10px
    pixel_thresh = 10
    dispMagnitude = torch.sqrt(torch.pow(d_gt[:,:,0],2) + torch.pow(d_gt[:,:,1], 2)).unsqueeze(-1).expand(-1,-1,2)
    idx = dispMagnitude > pixel_thresh
    z = torch.zeros(dispMagnitude.shape)
    d = torch.where(idx, z, d)
    d_gt = torch.where(idx, z, d_gt)

    # Calculate loss according to formula in paper
    return torch.sum(torch.sqrt(torch.diagonal(torch.bmm(d - d_gt, (d-d_gt).permute(0,2,1)), dim1=-2, dim2=-1)), dim = 1)


def get_mask(height, width, grid_size = 10):
    """
    Get the location based on the image size corresponding to relu_4_2
    and relu_5_1 layer for a desired grid size.
    """
    print(height, width)
    x_jump = int(width/grid_size)
    y_jump = int(height/grid_size)
    x_idx = np.linspace(int(x_jump/2),int(width - x_jump/2), grid_size, dtype = np.int32)
    y_idx = np.linspace(int(y_jump/2), int(height - y_jump/2), grid_size, dtype = np.int32)
    f_mask = torch.zeros((height//(2**4),width//2**4)).byte()
    u_mask = torch.zeros((height//(2**3),width//2**3)).byte()
    for i in x_idx:
        for j in y_idx:
            f_mask[j//(2**4),i//(2**4)] = 1
            u_mask[j//(2**3),i//(2**3)] = 1
    return(u_mask, f_mask)

"""

def test(args, epoch, model, data_loader):
        total_loss = 0

        model.eval()
        title = 'Validating Epoch {}'.format(epoch)
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=len(data_loader), smoothing=.9, miniters=1, leave=True, desc=title)
        predictions = []
        gt = []
        pck = []

        sys.stdout.flush()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(progress):
                target = target.squeeze()
                for i in range(0, data[0].shape[1], 150):
                    d = model(data[0][0,i:i+150].squeeze().to(args.device), im_2 = data[1][0, i:i+150].squeeze().to(args.device))
                    loss = _apply_loss(d, target[i:i+150]).sum()
                    total_loss += loss.item()
                    predictions.extend(d.numpy())

                title = '{} Epoch {}'.format('Validating', epoch)
                pck_sample = tools.calc_pck(np.asarray(predictions), np.asarray(gt))
                pck.append(pck_sample)

                sys.stdout.flush()


        progress.close()

        print('\tPCK for epoch %d is %f'%(epoch, pck.mean()))

        return total_loss / float(batch_idx + 1), pck.mean()
"""
def test(args, epoch, model, data_loader):
        statistics = []
        total_loss = 0

        model.eval()
        title = 'Validating Epoch {}'.format(epoch)
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=len(data_loader), smoothing=.9, miniters=1, leave=True, desc=title)
        predictions = []
        gt = []

        sys.stdout.flush()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(progress):

                d = model(data[0].to(args.device), im_2 = data[1].to(args.device))
                loss = _apply_loss(d, target).mean()
                total_loss += loss.item()
                predictions.extend(d.numpy())
                gt.extend(target.numpy())

                # Print out statistics
                statistics.append(loss.item())
                title = '{} Epoch {}'.format('Validating', epoch)

                progress.set_description(title + '\tLoss:\t'+ str(statistics[-1]))
                sys.stdout.flush()


        progress.close()
        pck = tools.calc_pck(np.asarray(predictions), np.asarray(gt))
        print('PCK for epoch %d is %f'%(epoch, pck))

        return total_loss / float(batch_idx + 1), pck

def train(args, epoch, model, data_loader, optimizer):
        statistics = []
        total_loss = 0

        model.train()
        title = 'Training Epoch {}'.format(epoch)
        progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=len(data_loader), smoothing=.9, miniters=1, leave=True, desc=title)

        sys.stdout.flush()

        for batch_idx, (data, target) in enumerate(progress):

            #data, target = data.to(args.device), target.to(args.device)

            optimizer.zero_grad()
            d = model(data[0].to(args.device), im_2 = data[1].to(args.device))
            loss = _apply_loss(d, target).mean()



            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += loss.item()
            assert not np.isnan(total_loss)

            # Print out statistics
            statistics.append(loss.item())
            title = '{} Epoch {}'.format('Training', epoch)

            progress.set_description(title + '\tLoss:\t'+ str(statistics[-1]))
            sys.stdout.flush()


        progress.close()

        return total_loss / float(batch_idx + 1)


if __name__ == '__main__':
    args = init_config()
    if not args.eval:
        sys.stdout = Logger(args.log_path)


    gpuargs = {'num_workers': args.effective_number_workers,
               'pin_memory': True,
               'drop_last' : True} if args.cuda else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    if args.dataset.lower() == 'sintel':
        train_dataset = MpiSintelFinal(os.path.join(args.data_path', sintel/training'), transforms = transforms)
        val_dataset = MpiSintelFinal(os.path.join(args.data_path)'sintel/training'), train = False, sequence_list = train_dataset.sequence_list, transforms = transforms)
        #test_dataset = MpiSintelFinal('/cluster/scratch/ninos/3dv/data/sintel/test')
    else:
        raise Exception('Dataset not supported yet.')
        sys.stdout.flush()


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size*torch.cuda.device_count(), shuffle=True, **gpuargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_test*torch.cuda.device_count(), shuffle=False, **gpuargs)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, **gpuargs)

    model = VGG_graph_matching()

    if args.use_vgg:
        model.copy_params_from_vgg16()

    if torch.cuda.device_count() > 1 and args.number_gpus > 1:
        model = nn.DataParallel(model)
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    best_pck = 0.
    for i in range(1, args.n_epochs+1):
        train(args, i, model, train_loader, optimizer)
        loss, pck = test(args, i, model, val_loader)
        if pck > best_pck:
            best_pck = pck
            torch.save(model.state_dict(), args.save_path)

