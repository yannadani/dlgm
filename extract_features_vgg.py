import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import pickle as pkl

vgg16 = models.vgg16_bn(pretrained=True)
vgg16.classifier = nn.Linear(25088, 1)
vgg16.eval()
#outputs will hold intermediate features from the vgg-16 network wherever the hook(s) are attached to.
global outputs
def hook(model, input, output):
    outputs.append(output.squeeze().detach().numpy())

transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ])
relu4_1_hook = vgg16.features[26].register_forward_hook(hook)
relu4_2_hook = vgg16.features[29].register_forward_hook(hook)
relu4_3_hook = vgg16.features[32].register_forward_hook(hook)
relu5_1_hook = vgg16.features[36].register_forward_hook(hook)
relu5_2_hook = vgg16.features[39].register_forward_hook(hook)
relu5_3_hook = vgg16.features[42].register_forward_hook(hook)

im_path = 'data/middlebury/images/'
out_path = 'data/middlebury/features/'
for root,dirs,files in os.walk(im_path):
    if not root.split('/')[-1] == 'images':
        if not osp.exists(osp.join(out_path,root.split('/')[-1])):
            os.makedirs(osp.join(out_path, root.split('/')[-1]))
    for i in files:
        if i.startswith('.'):
            continue

        outputs = []
    #Create a grid of pixels
        im = Image.open(osp.join(root,i)).convert('RGB')
        width, height = im.size
        x_jump = int(width/5)
        y_jump = int(height/5)
        x_idx = np.arange(int(x_jump/2),int(width - x_jump/2), x_jump)
        y_idx = np.arange(int(y_jump/2), int(height - y_jump/2), y_jump)
        vgg16(transformation(im).unsqueeze_(0))
        F = np.zeros((512*3,100))
        U = np.zeros((512*3,100))
        print(outputs[3].shape)
        count = 0
        for k in x_idx:
            for j in y_idx:
                F[:, count] = np.concatenate((outputs[3][:,int(j/(2**4)),int(k/(2**4))],outputs[4][:,int(j/(2**4)),int(k/(2**4))],outputs[5][:,int(j/(2**4)),int(k/(2**4))]))
                U[:, count] = np.concatenate((outputs[0][:,int(j/(2**3)),int(k/(2**3))],outputs[1][:,int(j/(2**3)),int(k/(2**3))],outputs[2][:,int(j/(2**3)),int(k/(2**3))]))
                count += 1
        with open(osp.join(out_path,root.split('/')[-1],i.split('.')[0]+'.pkl'),'wb') as write:
            pkl.dump({'F':F,'U':U},write)
