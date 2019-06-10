"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
#from data import create_dataset
from cycle_gan import cycleGAN 
import opt
import dataloader
import torch
from torch.utils.data import DataLoader
#from util import visualizer

if __name__ == '__main__':
       # get training options
       #dataset_total = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    batch_size = 1
    
    dataset_path = './dataset/landscape'
    landscape_set = dataloader.GANTransDataset(dataset_path, mode = 'flower')
    style_set = dataloader.GANTransDataset(dataset_path, mode = 'oilpaint')
    dataset = dataloader.GANCombinedDataset(landscape_set, style_set)# NEW CODE LINE
    landscape_size = len(landscape_set)    # get the number of images in the dataset.
    style_size = len(style_set)
    dataset_size = len(dataset)# NEW CODE LINE
    print('dataset size = %d' %dataset_size)# NEW CODE LINE
    print('The number of training images = %d' % landscape_size)
    print('The number of style images = %d' % style_size)

    model = cycleGAN()      # create a model given opt.model and other options  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(device)
    model.to(device)   
    #model.setup(opt)               # regular setup: load and print networks; create schedulers
    #写一可视化的方法visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    # create dataloader 
    dataset_loader = DataLoader(dataset, batch_size= batch_size, shuffle=True) # NEW CODE LINE
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, (real_A,real_B) in enumerate(dataset_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(real_A.to(device),real_B.to(device))         # unpack data from dataset and apply preprocessing
            model.forward()
            model.optimize_step()
  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                #model.compute_visuals()
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                #print()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                #visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                #if opt.display_id > 0:
                #    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
