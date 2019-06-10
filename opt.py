# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:30:27 2019

@author: shiyu
"""
dataroot = './dataset/landscape'
batch_size = 1
serial_batches = False
num_threads = 4
max_dataset_size = 10000
dataset_mode = 'unaligned' #[unaligned | aligned | single]
phase = 'train' # train, val, test, etc
resize_or_crop = 'resize_and_crop' # [resize_and_crop|crop|scale_width|scale_width_and_crop|none]
loadSize = 440
fineSize = 396 # crop images to this
output_nc = 3
input_nc = 3
isTrain = True # True or test
direction = 'AtoB'
no_flip = False
# no_flip: if specified, do not flip the images for data augmentation

# General
model = 'cycle_gan'
no_lsgan = True
gpu_ids = [0] #e.g. 0  0,1,2, 0,2. use -1 for CPU'

# Training
name = 'experiment'
checkpoints_dir = './checkpoints'
suffix = '' # customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}
epoch = 'latest'
load_iter = 0
continue_train = False
save_latest_freq = 100 # frequency of saving the latest results
epoch_count = 1
lambda_A = 10.0
lambda_B = 10.0
save_by_iter = True
niter = 50  # of iter at starting learning rate
niter_decay = 50 # of iter to linearly decay learning rate to zero
lambda_identity = 0.5
channels = 3
out_channels = 3
n_residual_blocks = 9
image_size = (256,256)
lr = 0.0002
b1 = 0.5
b2 = 0.999
device = 'cuda'
save_dir = "./saved_models/"


# CGAN
'''
norm = 'instance' 
netG = 'resnet_9blocks'
netD = 'basic'
ngf = 64 # gen filters in first conv layer
ndf = 64 # of discrim filters in first conv layer
init_type = 'xavier' # [normal|xavier|kaiming|orthogonal]
init_gain = 0.02 # 'scaling factor for normal, xavier and orthogonal.'
no_dropout = True
n_layers_D = 3
lr = 0.0002 # Learning rate adam
beta1 = 0.5 # momentum term of adam
pool_size = 50 # the size of image buffer that stores previously generated images
lr_policy = 'lambda'
lr_decay_iters = 50  # multiply by a gamma every lr_decay_iters iterations
'''

# Visiualization
display_id = 1 #window id of the web display
display_winsize = 256
display_ncols = 4 # display all images in a single visdom web panel with certain number of images per row
display_server = "http://localhost"
display_env = 'main'
display_port = 8097
update_html_freq = 1000 # frequency of saving training results to html
no_html = True
display_freq = 400;
print_freq = 100;
save_epoch_freq = 50;