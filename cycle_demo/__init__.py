import os
import opt
from cycle_gan import cycleGAN
import dataloader
import torch
from train import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
def load_trained_model(photo = 'landscape', style = 'picasso', epo = 'latest'):
    print('Loading photo = ',photo, 'and style = ',style)
    model = cycleGAN()     # create a model given opt.model and other options
    model.save_dir = f'./saved_models_{photo}_{style}/'
    model.load_networks(epo)
    return model
 

def load_dataset(dataset_path = './dataset', photo = 'landscape', style = 'picasso'):
    landscape_set = dataloader.GANTransDataset(dataset_path, mode = photo)
    style_set = dataloader.GANTransDataset(dataset_path, mode = style)
    dataset = dataloader.GANCombinedDataset(landscape_set, style_set)
    dataset_loader = DataLoader(dataset, batch_size= 1, shuffle=False)
    return dataset_loader
    
def torchimshow(image, ax=None):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    
def plot_result(model):
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12,8))
    torchimshow(model.real_A[0],ax=axes[0][0])
    axes[0][0].set_title('real image  A')
    torchimshow(model.fake_B[0],ax=axes[0][1])
    axes[0][1].set_title('style transfered to B')
    torchimshow(model.rec_A[0],ax=axes[0][2])
    axes[0][2].set_title('recovered image A')
    torchimshow(model.real_B[0],ax=axes[1][0])
    axes[1][0].set_title('real image B')
    torchimshow(model.fake_A[0],ax=axes[1][1])
    axes[1][1].set_title('style transfered to A')
    torchimshow(model.rec_B[0],ax=axes[1][2])
    axes[1][2].set_title('recovered image B')
    return