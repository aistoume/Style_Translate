import time
from cycle_gan import cycleGAN 
import opt
import dataloader
import torch
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
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

    model = cycleGAN(opt)      # create a model given opt.model and other options  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)
    model.load_networks(opt.epoch)
    Loss = model.load_loss(opt.epoch)
    # create dataloader 
    dataset_loader = DataLoader(dataset, batch_size= batch_size, shuffle=True) # NEW CODE LINE
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        loss_list = torch.zeros(8)
        for i, (real_A,real_B) in enumerate(dataset_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            epoch_iter += opt.batch_size
            model.set_input(real_A.to(device),real_B.to(device))         # unpack data from dataset and apply preprocessing
            model.forward()
            model.optimize_step()
            loss_list += torch.Tensor(model.return_loss())
        loss_list /= len(dataset)
        Loss.append(loss_list)


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            model.save_loss(Loss, 'latest')
            model.save_loss(Loss)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
