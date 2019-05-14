import argparse, os, sys, subprocess
from tqdm import tqdm
from glob import glob
from os.path import *


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pickle as pkl

from models import VGG_graph_matching
from dataloader import MpiSintelClean, MpiSintelFinal, ImagesFromFolder
from logger import Logger



def init_config():
    parser = argparse.ArgumentParser(description='Deep Learning of Graph matching')

    # dataset
    parser.add_argument('--dataset', type=str, default='sintel', help='dataset to use: midlebury/sintel')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='Perform Inference')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_flow', action='store_true', default=False, help='Save flow files during evaluation')

    # others
    parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')  
    parser.add_argument('--cuda', action='store_true', default=True, help='Use GPU') 



    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "models/%s" % args.dataset
    log_dir = "logs/%s" % args.dataset

    config_file = "config.config_%s" % args.dataset
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
    return torch.sum(torch.sqrt(torch.inner(d - d_gt, (d-d_gt).t()) + 1e-6))


def get_mask(width, height, grid_size = 10): 
    """
    Get the location based on the image size corresponding to relu_4_2 
    and relu_5_1 layer for a desired grid size. 
    """
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

def test(args, epoch, model, test_loader):
	model.eval()
        
        if args.save_flow:
            flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(args.save,args.name.replace('/', '.'),epoch)
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ', 
            leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
                data, target = data.to(args.device), target.to(args.device)

            with torch.no_grad():
                d = model(data[0], target[0], data[1], target[1], inference=True)
                loss = _apply_loss(d, target)

            losses = [torch.mean(loss_value) for loss_value in losses] 
            loss_val = losses[0] # Collect first loss for weight update
            total_loss += loss_val.item()
            loss_values = [v.item() for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            statistics.append(loss_values)
            
            if args.save_flow:
                for i in range(args.batch_size_test):
                    _pflow = output[i].item().numpy().transpose(1, 2, 0)
                    flow_utils.writeFlow( join(flow_folder, '%06d.flo'%(batch_idx * args.inference_batch_size + i)),  _pflow)

            progress.set_description('Inference Averages for Epoch {}: '.format(epoch) + tools.format_dictionary_of_losses(loss_labels, np.array(statistics).mean(axis=0)))
            progress.update(1)

        progress.close()
        sys.stdout.flush()

        return
 def train(args, epoch, model, data_loader, optimizer=None, is_validate=False):
        statistics = []
        total_loss = 0

        if is_validate:
            model.eval()
            title = 'Validating Epoch {}'.format(epoch)
            args.validation_n_batches = np.inf if args.validation_n_batches < 0 else args.validation_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=100, total=np.minimum(len(data_loader), args.validation_n_batches), leave=True, position=offset, desc=title)
        else:
            model.train()
            title = 'Training Epoch {}'.format(epoch)
            args.train_n_batches = np.inf if args.train_n_batches < 0 else args.train_n_batches
            progress = tqdm(tools.IteratorTimer(data_loader), ncols=120, total=np.minimum(len(data_loader), args.train_n_batches), smoothing=.9, miniters=1, leave=True, position=offset, desc=title)

        sys.stdout.flush()
        for batch_idx, (data, target) in enumerate(progress):

            data, target = data.to(args.device), target.to(args.device)

            optimizer.zero_grad() if not is_validate else None
            d = model(data[0], get_mask(data[0]), data[1], get_mask(data[1]))
            loss = _apply_loss(d, target)
            total_loss += loss_val.item()
            loss_values = [v.item() for v in losses]


            assert not np.isnan(total_loss)

            if not is_validate:
                loss_val.backward()
                if args.gradient_clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)
                 optimizer.step()

            # Print out statistics
            statistics.append(loss_values)
            title = '{} Epoch {}'.format('Validating' if is_validate else 'Training', epoch)

            progress.set_description(title + ' ' + tools.format_dictionary_of_losses(loss_labels, statistics[-1]))
            sys.stdout.flush()


        progress.close()

        return total_loss / float(batch_idx + 1), (batch_idx + 1)


if __name__ == '__main__':
	args = init_config()
    if not args.eval:
        sys.stdout = Logger(args.log_path)


    gpuargs = {'num_workers': args.effective_number_workers, 
               'pin_memory': True, 
               'drop_last' : True} if args.cuda else {}

    if args.dataset == 'sintel':
		train_dataset = MpiSintelFinal('~/Downloads/MPI-Sintel-complete/training')
		val_dataset = MpiSintelClean('~/Downloads/MPI-Sintel-complete/training')
		test_dataset = MpiSintelClean('~/Downloads/MPI-Sintel-complete/training')

	elif args.dataset == 'midlebury':
		train_dataset = ImagesFromFolder('~/Downloads/MPI-Sintel-complete/training') #TODO: Change the root
		val_dataset = ImagesFromFolder('~/Downloads/MPI-Sintel-complete/training')
		test_dataset = ImagesFromFolder('~/Downloads/MPI-Sintel-complete/training')
	else:
		print('Dataset not supported. Choose between Midlebury and Sintel.')
		sys.stdout.flush()
		return

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **gpuargs)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size_test, shuffle=False, **gpuargs)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, **gpuargs)
