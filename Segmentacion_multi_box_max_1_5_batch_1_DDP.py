import time
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import torch
import torch.nn as nn

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate

import torch.multiprocessing as mp

from datasets.data_augmentation import ToTensor_mono,ToTensor_ms, RandomHorizontalFlip_ms, RandomVerticalFlip_ms, RandomTimeMirror, RandomEventDrop_ms

from OF_EV_SNN.network_3d.poolingNet_cat_1res import NeuronPool_Separable_Pool3d_2, \
                    PLIFNeuronPool_Separable_Pool3d, \
                    PLIFNeuronPool_Separable_Pool3d_2, \
                    NeuronPool_Separable_Pool3d, \
                    NeuronPool_Separable_Pool3d_2, \
                    PLIFNeuronPool_Separable_Pool3d_ts, \
                    NeuronPool_Separable_Pool3d_3x3,  \
                    PLIFNeuronPool_Separable_Pool3d_state      


from network.Params import yamlParams
from network.loss2 import Total_Loss_Motion, DiceLoss, BCEDiceLoss, getIOU
from datasets.MVSEC.mvsec_dataset import EVIMODatasetBase, EVIMODatasetBaseNPZ
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from viz import show_learning, show_loss
from PIL import Image
import os

from spikingjelly.clock_driven import neuron, layer
from torch._C import dtype
MultiStep=False

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

########################
# COMPLEMENT FUNCTIONS #
########################

# def my_collate(batch):
#     batch = list(filter (lambda x:x is not None, batch))
#     if not batch:
#         return None

#     return default_collate(batch)
    
##############################
# DEVICE AND REPRODUCIBILITY #
##############################

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# rank = 2
# torch.cuda.set_device(rank-1)

# logfile = open("./results/checkpoints/test_results_dani.txt", "w+")

###########################
# VISUALIZATION FUNCTIONS #
###########################

# plt.ion()
# fig = plt.figure()


############################
############################
##		  DATABASE		  ##
##	 	   PARAMS		  ##
############################
############################

######################
# GENERAL PARAMETERS #
######################
nfpdm = 1              # number of frames per depth map (1 label every 50 ms)
N_inference = 1        # number of chunks for training/testing (1 chunk = 50 ms = nfpdm frames)
N_warmup = 1           # number of chunks for warmup (if you want to use a stateful model)

# mode = "depth"		     #depth/motion
mode = "motion"		     #depth/motion
datasetType = "EVIMO"

##############################################
# SPIKEMS PARAMETERS ONLY FOR MOTION SEGMENT #
##############################################

# crop = False # True
# maxBackgroundRatio = 1.5 #relation background/foreground
# checkpoint = "/content/drive/MyDrive/SpikeMS/pretrainedModels/EVIMO-pretrained/out/checkpoint.pth.tar" 
# modeltype = "unetRNN6Layer_noBlock"

incrementalPercent  = 1
saveImages = True
imageLabel = ""

#output_dir = "/content/drive/MyDrive/StereoSpike/logs/" # os.path.join(os.getcwd(), 'logs')

# general_config = "general_config4.yaml"
# genconfigs = yamlParams(general_config)

########################
## TRAINING FRAMEWORK ##
########################
# Create the network

# torch.distributed.init_process_group(
#     backend='nccl', world_size=rank)
    
# model = PLIFNeuronPool_Separable_Pool3d(multiply_factor = 1.)#.to(device) #ts 20
# net = DDP(model, device_ids=[rank])#, output_device=i)

#net_not_DDP = PLIFNeuronPool_Separable_Pool3d(multiply_factor = 1.).to(device) #ts 20
# net = NeuronPool_Separable_Pool3d_3x3(multiply_factor = 10.).to(device)

# trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print('Trainable parameters: {}'.format(trainable_params))

#model = torch.nn.DataParallel(net_not_DDP)

# Initialize network weights

# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)

# learning_rate = 2e-3
# weight_decay = 0.0
# learned_metric = 'LIN'
# # Create the optimizer
# lr = 5e-4
# wd = 1e-3
# optimizer = torch.optim.AdamW(net.parameters(), lr = lr, weight_decay = wd)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20, 40], gamma = 0.5)

# loss_module = BCEDiceLoss()

############################
############################
##		MULTI-DATASET	  ##
##	   	 CONCATENED	      ##
############################
############################

#maskDir_test = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_00/img" 
#datafile_test = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_00.hdf5"
#maskDir_train = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_00/img" 
#datafile_train = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_00.hdf5"

#maskDir_test1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_01/img" 
#datafile_test1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_01.hdf5"
#maskDir_train1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_01/img" 
#datafile_train1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_01.hdf5"

#maskDir_test2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_02/img" 
#datafile_test2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_02.hdf5"
#maskDir_train2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_02/img" 
#datafile_train2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_02.hdf5"

#maskDir_test3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_03/img" 
#datafile_test3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_03.hdf5"
#maskDir_train3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_03/img" 
#datafile_train3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_03.hdf5"

#maskDir_test4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_04/img" 
#datafile_test4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_04.hdf5"
#maskDir_train4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_04/img" 
#datafile_train4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_04.hdf5"

#maskDir_test5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_05/img" 
#datafile_test5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_05.hdf5"
#maskDir_train5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_05/img" 
#datafile_train5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_05.hdf5"

#maskDir_train6 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_06/img" 
#datafile_train6 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_06.hdf5"

#maskDir_train7 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_07/img" 
#datafile_train7 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_07.hdf5"

#maskDir_train8 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_08/img" 
#datafile_train8 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_08.hdf5"

#maskDir_train9 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_09/img" 
#datafile_train9 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_09.hdf5"

#maskDir_train10 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_10/img" 
#datafile_train10 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_10.hdf5"

#maskDir_train11 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_11/img" 
#datafile_train11 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_11.hdf5"

##general_config = "general_config4.yaml"
#genconfigs = yamlParams(general_config)

#print("data_motion")
#tsfm = transforms.Compose([
    #RandomHorizontalFlip_ms(p=0.5),
    #RandomVerticalFlip_ms(p=0.5),
    #RandomEventDrop_ms(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
#])
#if(datasetType == "EVIMO"):
    #database_train = EVIMODatasetBase(datafile_train, genconfigs, maskDir_train, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_test = EVIMODatasetBase(datafile_test, genconfigs, maskDir_test, crop, 
                                #maxBackgroundRatio, incrementalPercent)
    #database_train1 = EVIMODatasetBase(datafile_train1, genconfigs, maskDir_train1, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_test1 = EVIMODatasetBase(datafile_test1, genconfigs, maskDir_test1, crop, 
                                #maxBackgroundRatio, incrementalPercent)
    #database_train2 = EVIMODatasetBase(datafile_train2, genconfigs, maskDir_train2, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_test2 = EVIMODatasetBase(datafile_test2, genconfigs, maskDir_test2, crop, 
                                #maxBackgroundRatio, incrementalPercent)
    #database_train3 = EVIMODatasetBase(datafile_train3, genconfigs, maskDir_train3, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_test3 = EVIMODatasetBase(datafile_test3, genconfigs, maskDir_test3, crop, 
                                #maxBackgroundRatio, incrementalPercent)
    #database_train4 = EVIMODatasetBase(datafile_train4, genconfigs, maskDir_train4, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_test4 = EVIMODatasetBase(datafile_test4, genconfigs, maskDir_test4, crop, 
                                #maxBackgroundRatio, incrementalPercent)
    #database_train5 = EVIMODatasetBase(datafile_train5, genconfigs, maskDir_train5, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_test5 = EVIMODatasetBase(datafile_test5, genconfigs, maskDir_test5, crop, 
                                #maxBackgroundRatio, incrementalPercent)
    #database_train6 = EVIMODatasetBase(datafile_train6, genconfigs, maskDir_train6, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_train7 = EVIMODatasetBase(datafile_train7, genconfigs, maskDir_train7, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_train8 = EVIMODatasetBase(datafile_train8, genconfigs, maskDir_train8, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_train9 = EVIMODatasetBase(datafile_train9, genconfigs, maskDir_train9, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_train10 = EVIMODatasetBase(datafile_train10, genconfigs, maskDir_train10, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    #database_train11 = EVIMODatasetBase(datafile_train11, genconfigs, maskDir_train11, crop, 
                                #maxBackgroundRatio, incrementalPercent,tsfm)
    
    #print("EVIMO used")
#else:
    #raise Exception("Only EVIMO dataset with hdf5 format generated by preprocessing scripts handled with this code.")

## uncomment if you want to split test/train using single hdf5 file
#torch.manual_seed(0)     
#test_split = genconfigs['model']['testSplit'] # self.genconfigs['model']['testSplit']                      #

#train_dev_sets = torch.utils.data.ConcatDataset([database_train, database_train1, database_train2, database_train3, database_train4, database_train5, database_train6, database_train7, database_train8, database_train9, database_train10, database_train11])
#test_dev_sets = torch.utils.data.ConcatDataset([database_test, database_test1, database_test2, database_test3, database_test4, database_test5])

## uncomment if you want to split test/train using single hdf5 file
#num_workers = genconfigs['hardware']['readerThreads']
#batch_size = genconfigs['batchsize']


#test_loader = torch.utils.data.DataLoader(
        #dataset=test_dev_sets,                                                       #
        #batch_size=batch_size,
        #shuffle=False,
        #num_workers=num_workers,
        #pin_memory=True,
        #collate_fn=my_collate,
        #drop_last=False)        

#train_loader = torch.utils.data.DataLoader(
        #dataset=train_dev_sets,
        #batch_size=batch_size,
        #shuffle=True,
        #num_workers=num_workers,
        #pin_memory=True,
        #collate_fn=my_collate,
        #drop_last=False)          

#def set_random_seed(seed):
    ## Python
    #random.seed(seed)

    ## PyTorch
    #torch.manual_seed(seed)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #if int(torch.__version__.split('.')[1]) < 8:
        #torch.set_deterministic(True)  # for pytorch < 1.8
    #else:
        #torch.use_deterministic_algorithms(True)

    ## NumPy
    #np.random.seed(seed)

############################
############################
##	   	 	START		  ##
##	    	TRAIN		  ##
############################
############################

#try:
  #print(epoch_load)
#except:
  #epoch_load = 0

#train_loss_vec = []
#train_error_vec = []
#test_loss_vec = []
#test_error_vec = []

#total_MSE = 0
#total_IOU = 0
#scalar_i = 0
#tot_frames = 0


#functional.reset_net(net)         
#n_epochs = 300


#for epoch in range(epoch_load+1,n_epochs,1):
    #train_i = 0
    #test_i = 0

    #show=False
    #net = net.train()                                                   #<--------   0
    #running_train_loss = 0
    #running_train_IOU = 0
    #start_time = time.time()    
    #with torch.autograd.set_detect_anomaly(True):
        #for i, data in enumerate(tqdm(train_loader, desc='training')):  #loader --> train_loader
            #if (data == None):
                ## print(len(train_loader))
                #continue
            #train_i += 1   
            #functional.reset_net(net)                                   #<--------  6                
            #spikes_input = data['spike_tensor']                         #   torch.Size([8, 2, 260, 346, 10]) (B,C,H,W,T)
            #spikes_masked = data['masked_spike_tensor']

            #print(" ")
            #print("spikes_input.shape: ", spikes_input.shape)

            #spikes_input = spikes_input.permute(0,1,4,2,3)              #   torch.Size([8, 2, 10, 260, 346])
            #spikes_masked = spikes_masked.permute(0,1,4,2,3)            # (0,1,4,2,3)  
            #if MultiStep:                                               #   sums all time steps in the dim=0
                #spikes_input  = torch.sum(spikes_input, axis=0,keepdim=True)
                #spikes_masked = torch.sum(spikes_masked,axis=0,keepdim=True)                                  
            #spikes_input = spikes_input.to(device)                      #   torch.Size([B,C,T,260, 346])
            #spikes_masked = spikes_masked.to(device)

            #spike_pred_list =  net(spikes_input)
            #spike_pred =  spike_pred_list[2].to(device)
            #spike_pred = torch.sigmoid(spike_pred)

            #if spike_pred.ndim == 4:
                #spike_input_crop  =  spikes_input[:, :, :, :spike_pred.size(2), :spike_pred.size(3)]       #ndim 4  (B,2,21,260,346)
                #spikes_masked_crop =spikes_masked[:, :, :, :spike_pred.size(2), :spike_pred.size(3)]       #ndim 4  (B,1,1,260,346)
                #spike_pred_4D = torch.sum(spike_pred, axis = (1),keepdim=True) #
                
            #elif spike_pred.ndim == 5:
                #spike_pred_4D = torch.sum(spike_pred, axis = (2))               #torch.Size([B, 1, 260, 346]) 
                #spike_input_crop  =  spikes_input[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]       #ndim 5
                #spikes_masked_crop =spikes_masked[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]       #ndim 5
            
            #spike_mask_4D = torch.any(spikes_masked_crop, dim = 2,keepdim=False) #or and remove of TS dim 
            #spike_mask_4D = torch.any(spike_mask_4D, dim = 1,keepdim=True)      #or of C dim
            #spike_mask_4D = torch.mul(spike_mask_4D,1.)          #conversion to 
            

            #if show:
                    ##condition to colapse all timesteps in every batch
                #spike_input_5D = torch.sum(spikes_input, axis = (2),keepdim=True) 
                #spike_input_5D = spike_input_5D.permute(0,2,1,3,4)      #  (B,1,C,H,W)
                
                #for e in range(spike_input_5D.size(0)):
                    #show_learning(fig,False, 
                                  #spike_input_5D[None,e,:,:,:,:], 
                                  #spike_pred_4D[None,e,:,:,:], 
                                  #spike_mask_4D[None,e,:,:,:], 'segm, batch '+str(e),i)                        
            #loss = loss_module(spike_pred_4D, spike_mask_4D,weightbce=0.1)        #<--------  3
            #loss.backward()                                         #<--------  4
            #optimizer.step()                                        #<--------  5

            #optimizer.zero_grad()                               #<--------   1 
            #running_train_loss += loss.item() / spikes_input.size(0)
            #spike_pred = torch.round(spike_pred)
            #running_train_IOU += getIOU(spike_pred,spike_mask_4D)

                ## process saved metrics
        #if train_i != 0:        
            #epoch_train_loss = running_train_loss / train_i # len(train_loader)
            #print("\n train i: ",train_i)
            #print("len train loader: ", len(train_loader))
            #epoch_train_IOU = running_train_IOU / train_i #len(train_loader)
            #train_loss_vec.append(float(epoch_train_loss))
            #train_error_vec.append(float(epoch_train_IOU))
            #epoch_train_time = time.time() - start_time
            #train_epoch_summary = "Epoch: {}, Train Loss: {}, IOU Error (m): {}, Time: {}\n".format(epoch,epoch_train_loss,epoch_train_IOU,epoch_train_time)
        
##################
    #running_test_loss = 0
    #running_test_IOU = 0
    #net.eval()
    #show=False
    
    #with torch.no_grad():
        #start_time = time.time()
        #for i, data in enumerate(tqdm(test_loader, desc='testing')):  #loader --> train_loader
            #if (data == None):
                #continue
            #test_i += 1    
            #functional.reset_net(net)     
            #optimizer.zero_grad()        
            #spikes_input = data['spike_tensor']                 #   torch.Size([8, 2, 260, 346, 21]) (B,C,H,W,T)
            #spikes_masked = data['masked_spike_tensor']
            #spikes_input = spikes_input.permute(0,1,4,2,3)     #   torch.Size([8, 2, 21, 260, 346])
            #spikes_masked = spikes_masked.permute(0,1,4,2,3)  
            #spikes_input = spikes_input.to(device)              #   torch.Size([B,C,T,260, 346])
            #spikes_masked = spikes_masked.to(device)
            #spikes_input.requires_grad_()
            #spike_pred_list =  net(spikes_input)                # len(spike_pred_list) = 4
            #spike_pred =  spike_pred_list[2].to(device)             # (1,1,260,346)
            #spike_pred = torch.sigmoid(spike_pred)

            #if spike_pred.ndim == 4:
                #spike_input_crop = spikes_input[:,:, :, :spike_pred.size(2), :spike_pred.size(3)]       #ndim 4 (B,2,21,260,346)
                #spikes_masked_crop = spikes_masked[:,:, :, :spike_pred.size(2), :spike_pred.size(3)]    #ndim 4
                #spike_pred_4D = torch.sum(spike_pred, axis = (1),keepdim=True) #

            #elif spike_pred.ndim == 5:
                #spike_pred_4D = torch.sum(spike_pred, axis = (2))           #torch.Size([B, 1, 260, 346]) 
                #spike_input_crop = spikes_input[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]       #ndim 5
                #spikes_masked_crop = spikes_masked[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]    #ndim 5

            #spike_mask_4D = torch.any(spikes_masked_crop, dim = 2,keepdim=False) #or and reduce of TS 
            #spike_mask_4D = torch.any(spike_mask_4D, dim = 1,keepdim=True)      #or of channels
            #spike_mask_4D = torch.mul(spike_mask_4D,1.)          #conversion to         

            #if show:
                ##condition to colapse all timesteps in every batch
                ## print("spike_input: ", spikes_input.shape)
                #spike_input_5D = torch.sum(spikes_input, axis = (2),keepdim=True)       #  (B,C,T,H,W)-->(B,C,1,H,W)
                #spike_input_5D = spike_input_5D.permute(0,2,1,3,4)      #  (B,1,C,H,W)
                #for e in range(spike_input_5D.size(0)):
                    #show_learning(fig,False, 
                                  #spike_input_5D[None,e,:,:,:,:], 
                                  #spike_pred_4D[None,e,:,:,:], 
                                  #spike_mask_4D[None,e,:,:,:], 'segm, batch '+str(e),i)  #plt.fig,Save,[B,1,2,260,346],  [B, 1, 260, 346], [B, 1, 260, 346], text,       i
                ##show_learning(fig,False, spike_input_5D, spike_pred_4D, spike_mask_4D, 'segmentacion',i)  
            
            #loss = loss_module(spike_pred_4D, spike_mask_4D,weightbce=0.1)      
                
            #running_test_loss += loss / spikes_input.size(0)
            #spike_pred = torch.round(spike_pred)
            #running_test_IOU += getIOU(spike_pred, spike_mask_4D)

                ## process saved metrics
        #if test_i != 0:           
            #epoch_test_loss = running_test_loss / test_i #len(test_loader)        
            #epoch_test_IOU = running_test_IOU / test_i #len(test_loader)
            #print("\n test i: ",test_i)
            #print("len test loader: ", len(test_loader))
            #test_loss_vec.append(float(epoch_test_loss))
            #test_error_vec.append(float(epoch_test_IOU))
            #epoch_test_time = time.time() - start_time
            #test_epoch_summary = "Epoch: {}, Test Loss: {}, Test IOU Error (m): {}, Time: {}\n".format(epoch, epoch_test_loss,epoch_test_IOU,epoch_test_time)
            #print(train_epoch_summary + test_epoch_summary)
            #logfile.write(train_epoch_summary + test_epoch_summary)

        #if epoch_test_loss < net.get_max_accuracy():
            #print("Best performances so far: saving model...\n")
            #logfile.write("Best performances so far: saving model...\n")
            #net.update_max_accuracy(epoch_test_loss)
            ##torch.save(net.state_dict(), "./results/checkpoints/stereospike_seg_sergio.pth")
            #torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(),
                        #'optimizer_state_dict': optimizer.state_dict(),
                        #'loss': epoch_test_loss, 'train_loss_vec_dict': train_loss_vec,
                        #'test_loss_vec_dict': test_loss_vec,
                        #'train_error_vec_dict': train_error_vec,
                        #'test_error_vec_dict': test_error_vec
                        ## }, "./results/checkpoints/stereospike_seg_sergio4.pth")
                        #}, "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/results/checkpoints/poolingnet_seg_lif3x3_box_max1_5_batch4.pth")
            
        #net.increment_epoch()
        #scheduler.step()

#print("training finished !")

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gpu_id: int,
        save_every: int,
        loss_module,
    ) -> None:
        self.gpu_id = gpu_id
        self.net = model.to(gpu_id)
        self.train_loader = train_data
        self.test_loader = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.net = DDP(model, device_ids=[gpu_id])
        self.loss_module = loss_module
        
    def train(self, n_epochs: int):
      try:
        print(epoch_load)
      except:
        epoch_load = 0

      train_loss_vec = []
      train_error_vec = []
      test_loss_vec = []
      test_error_vec = []

      plt.ion()
      fig = plt.figure()

      total_MSE = 0
      total_IOU = 0
      scalar_i = 0
      tot_frames = 0

      best_accuracy = 10000.00
      # spikes_masked_crop = torch.zeros(2, 2, 2, 2)
      # spike_pred_4D = torch.zeros(2, 2, 2, 2)


      functional.reset_net(self.net)         
      #n_epochs = 300


      for epoch in range(0,n_epochs,1):
        train_i = 0
        test_i = 0

        show = True #False
        self.net.train()                                                   #<--------   0
        running_train_loss = 0
        running_train_IOU = 0
        start_time = time.time()    
        with torch.autograd.set_detect_anomaly(True):
          for i, data in enumerate(tqdm(self.train_loader, desc='training')):  #loader --> train_loader
            if (data == None):
              # print(len(train_loader))
              continue
            train_i += 1   
            functional.reset_net(self.net)                                   #<--------  6                
            spikes_input = data['spike_tensor']                         #   torch.Size([8, 2, 260, 346, 10]) (B,C,H,W,T)
            spikes_masked = data['masked_spike_tensor']

            # print(" ")
            # print("spikes_input.shape: ", spikes_input.shape)

            spikes_input = spikes_input.permute(0,1,4,2,3)              #   torch.Size([8, 2, 10, 260, 346])
            spikes_masked = spikes_masked.permute(0,1,4,2,3)            # (0,1,4,2,3)  
            if MultiStep:                                               #   sums all time steps in the dim=0
              spikes_input  = torch.sum(spikes_input, axis=0,keepdim=True)
              spikes_masked = torch.sum(spikes_masked,axis=0,keepdim=True)                                  
            spikes_input = spikes_input.to(self.gpu_id)                      #   torch.Size([B,C,T,260, 346])
            spikes_masked = spikes_masked.to(self.gpu_id)

            spike_pred_list =  self.net(spikes_input)
            spike_pred =  spike_pred_list[0].to(self.gpu_id)
            # print("spike_pred: ", spike_pred.shape)
            spike_pred = torch.sigmoid(spike_pred)
            spike_pred = spike_pred[None,:,:,:]
            # print("spike_pred1: ", spike_pred.shape)

            if spike_pred.ndim == 4:
              spike_input_crop  =  spikes_input[:, :, :, :spike_pred.size(2), :spike_pred.size(3)]       #ndim 4  (B,2,21,260,346)
              spikes_masked_crop = spikes_masked[:, :, :, :spike_pred.size(2), :spike_pred.size(3)]       #ndim 4  (B,1,1,260,346)
              spike_pred_4D = torch.sum(spike_pred, axis = (1),keepdim=True) #
              
            elif spike_pred.ndim == 5:
              spike_pred_4D = torch.sum(spike_pred, axis = (2))               #torch.Size([B, 1, 260, 346]) 
              spike_input_crop  =  spikes_input[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]       #ndim 5
              spikes_masked_crop =spikes_masked[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]       #ndim 5
            
            # print("spikes_masked_crop: ", spikes_masked_crop.shape)
            spike_mask_4D = torch.any(spikes_masked_crop, dim = 2,keepdim=False) #or and remove of TS dim 
            spike_mask_4D = torch.any(spike_mask_4D, dim = 1,keepdim=True)      #or of C dim
            spike_mask_4D = torch.mul(spike_mask_4D,1.)          #conversion to 
            

            # if show:
            #     #condition to colapse all timesteps in every batch
            #   spike_input_5D = torch.sum(spikes_input, axis = (2),keepdim=True) 
            #   spike_input_5D = spike_input_5D.permute(0,2,1,3,4)      #  (B,1,C,H,W)
              
            #   for e in range(spike_input_5D.size(0)):
            #     show_learning(fig,False, 
            #             spike_input_5D[None,e,:,:,:,:], 
            #             spike_pred_4D[None,e,:,:,:], 
            #             spike_mask_4D[None,e,:,:,:], 'segm, batch '+str(e),i)                        
            # print("spike_pred_4D.shape", spike_pred_4D.shape)
            # print("spike_mask_4D.shape", spike_mask_4D.shape)
            loss = self.loss_module(spike_pred_4D, spike_mask_4D,weightbce=0.1)        #<--------  3
            loss.backward()                                         #<--------  4
            self.optimizer.step()                                        #<--------  5

            self.optimizer.zero_grad()                               #<--------   1 
            running_train_loss += loss.item() / spikes_input.size(0)
            spike_pred = torch.round(spike_pred)
            running_train_IOU += getIOU(spike_pred,spike_mask_4D)

          
          spike_input_5D = torch.sum(spikes_input, axis = (2),keepdim=True) 
          spike_input_5D = spike_input_5D.permute(0,2,1,3,4)      #  (B,1,C,H,W)

          for e in range(spike_input_5D.size(0)):
            show_learning(fig,True, 
                    spike_input_5D[None,e,:,:,:,:], 
                    spike_pred_4D[None,e,:,:,:], 
                    spike_mask_4D[None,e,:,:,:], 'segm, batch '+str(e),i)

              # process saved metrics
          if train_i != 0:        
            epoch_train_loss = running_train_loss / train_i # len(train_loader)
            print("\n train i: ",train_i)
            print("len train loader: ", len(self.train_loader))
            epoch_train_IOU = running_train_IOU / train_i #len(train_loader)
            train_loss_vec.append(float(epoch_train_loss))
            train_error_vec.append(float(epoch_train_IOU))
            epoch_train_time = time.time() - start_time
            train_epoch_summary = "Epoch: {}, Train Loss: {}, IOU Error (m): {}, Time: {}\n".format(epoch,epoch_train_loss,epoch_train_IOU,epoch_train_time)
          
      #################
        running_test_loss = 0
        running_test_IOU = 0
        self.net.eval()
        show=False
        
        with torch.no_grad():
          start_time = time.time()
          for i, data in enumerate(tqdm(self.test_loader, desc='testing')):  #loader --> train_loader
            if (data == None):
              continue
            test_i += 1    
            functional.reset_net(self.net)     
            self.optimizer.zero_grad()        
            spikes_input = data['spike_tensor']                 #   torch.Size([8, 2, 260, 346, 21]) (B,C,H,W,T)
            spikes_masked = data['masked_spike_tensor']
            spikes_input = spikes_input.permute(0,1,4,2,3)     #   torch.Size([8, 2, 21, 260, 346])
            spikes_masked = spikes_masked.permute(0,1,4,2,3)  
            spikes_input = spikes_input.to(self.gpu_id)              #   torch.Size([B,C,T,260, 346])
            spikes_masked = spikes_masked.to(self.gpu_id)
            # spikes_input.requires_grad_()
            spike_pred_list =  self.net(spikes_input)                # len(spike_pred_list) = 4
            spike_pred =  spike_pred_list[0].to(self.gpu_id)             # (1,1,260,346) spike_pred_list[2].to(self.gpu_id) 
            spike_pred = torch.sigmoid(spike_pred)
            spike_pred = spike_pred[None,:,:,:]

            if spike_pred.ndim == 4:
              spike_input_crop = spikes_input[:,:, :, :spike_pred.size(2), :spike_pred.size(3)]       #ndim 4 (B,2,21,260,346)
              spikes_masked_crop = spikes_masked[:,:, :, :spike_pred.size(2), :spike_pred.size(3)]    #ndim 4
              spike_pred_4D = torch.sum(spike_pred, axis = (1),keepdim=True) #

            elif spike_pred.ndim == 5:
              spike_pred_4D = torch.sum(spike_pred, axis = (2))           #torch.Size([B, 1, 260, 346]) 
              spike_input_crop = spikes_input[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]       #ndim 5
              spikes_masked_crop = spikes_masked[:spike_pred.size(0), :spike_pred.size(1), :spike_pred.size(2), :spike_pred.size(3),:spike_pred.size(4)]    #ndim 5

            spike_mask_4D = torch.any(spikes_masked_crop, dim = 2,keepdim=False) #or and reduce of TS 
            spike_mask_4D = torch.any(spike_mask_4D, dim = 1,keepdim=True)      #or of channels
            spike_mask_4D = torch.mul(spike_mask_4D,1.)          #conversion to         

            # if show:
            #   #condition to colapse all timesteps in every batch
            #   # print("spike_input: ", spikes_input.shape)
            #   spike_input_5D = torch.sum(spikes_input, axis = (2),keepdim=True)       #  (B,C,T,H,W)-->(B,C,1,H,W)
            #   spike_input_5D = spike_input_5D.permute(0,2,1,3,4)      #  (B,1,C,H,W)
            #   # for e in range(spike_input_5D.size(0)):
            #     # show_learning(fig,False, 
            #     #         spike_input_5D[None,e,:,:,:,:], 
            #     #         spike_pred_4D[None,e,:,:,:], 
            #     #         spike_mask_4D[None,e,:,:,:], 'segm, batch '+str(e),i)  #plt.fig,Save,[B,1,2,260,346],  [B, 1, 260, 346], [B, 1, 260, 346], text,       i
            #   #show_learning(fig,False, spike_input_5D, spike_pred_4D, spike_mask_4D, 'segmentacion',i)  
            
            loss = self.loss_module(spike_pred_4D, spike_mask_4D,weightbce=0.1)      
              
            running_test_loss += loss / spikes_input.size(0)
            spike_pred = torch.round(spike_pred)
            running_test_IOU += getIOU(spike_pred, spike_mask_4D)

              # process saved metrics
          if test_i != 0:           
            epoch_test_loss = running_test_loss / test_i #len(test_loader)        
            epoch_test_IOU = running_test_IOU / test_i #len(test_loader)
            print("\n test i: ",test_i)
            print("len test loader: ", len(self.test_loader))
            test_loss_vec.append(float(epoch_test_loss))
            test_error_vec.append(float(epoch_test_IOU))
            epoch_test_time = time.time() - start_time
            test_epoch_summary = "Epoch: {}, Test Loss: {}, Test IOU Error (m): {}, Time: {}\n".format(epoch, epoch_test_loss,epoch_test_IOU,epoch_test_time)
            print(train_epoch_summary + test_epoch_summary)
            # logfile.write(train_epoch_summary + test_epoch_summary)

          if epoch_test_loss < best_accuracy:
            print("Best performances so far: saving model...\n")
            # logfile.write("Best performances so far: saving model...\n")
            best_accuracy = epoch_test_loss
            # self.net.update_max_accuracy(epoch_test_loss)
            #torch.save(net.state_dict(), "./results/checkpoints/stereospike_seg_sergio.pth")
            torch.save({'epoch': epoch, 'model_state_dict': self.net.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'loss': epoch_test_loss, 'train_loss_vec_dict': train_loss_vec,
                  'test_loss_vec_dict': test_loss_vec,
                  'train_error_vec_dict': train_error_vec,
                  'test_error_vec_dict': test_error_vec
                  # }, "./results/checkpoints/stereospike_seg_sergio4.pth")
                  }, "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/results/checkpoints/poolingnet_seg_lif3x3_box_max1_5_batch4.pth")
            
          #self.train_data.sampler.set_epoch(epoch)
          # self.net.increment_epoch()
          self.scheduler.step()



def dataSet(maxBackgroundRatio):
	# maskDir_test = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_00/img" 
	# datafile_test = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_00.hdf5"
	# maskDir_train = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_00/img" 
	# datafile_train = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_00.hdf5"
	
	# maskDir_test1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_01/img" 
	# datafile_test1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_01.hdf5"
	# maskDir_train1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_01/img" 
	# datafile_train1 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_01.hdf5"
	
	# maskDir_test2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_02/img" 
	# datafile_test2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_02.hdf5"
	# maskDir_train2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_02/img" 
	# datafile_train2 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_02.hdf5"
	
	# maskDir_test3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_03/img" 
	# datafile_test3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_03.hdf5"
	# maskDir_train3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_03/img" 
	# datafile_train3 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_03.hdf5"
	
	# maskDir_test4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_04/img" 
	# datafile_test4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_04.hdf5"
	# maskDir_train4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_04/img" 
	# datafile_train4 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_04.hdf5"
	
	# maskDir_test5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_05/img" 
	# datafile_test5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/eval/box/txt/seq_05.hdf5"
	# maskDir_train5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_05/img" 
	# datafile_train5 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_05.hdf5"
	
	# maskDir_train6 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_06/img" 
	# datafile_train6 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_06.hdf5"
	
	# maskDir_train7 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_07/img" 
	# datafile_train7 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_07.hdf5"
	
	# maskDir_train8 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_08/img" 
	# datafile_train8 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_08.hdf5"
	
	# maskDir_train9 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_09/img" 
	# datafile_train9 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_09.hdf5"
	
	# maskDir_train10 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_10/img" 
	# datafile_train10 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_10.hdf5"
	
	# maskDir_train11 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_11/img" 
	# datafile_train11 = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/datasets/EVIMO_Dataset/train/box/txt/seq_11.hdf5"
	
	
	# general_config = "/home/aircv1/Data/Gerardo/Daniela/StereoSpike/general_config1.yaml"
  # /content/drive/MyDrive/

	maskDir_test = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_00/img" 
	datafile_test = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_00.hdf5"
	maskDir_train = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_00/img" 
	datafile_train = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_00.hdf5"
	
	maskDir_test1 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_01/img" 
	datafile_test1 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_01.hdf5"
	maskDir_train1 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_01/img" 
	datafile_train1 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_01.hdf5"
	
	maskDir_test2 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_02/img" 
	datafile_test2 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_02.hdf5"
	maskDir_train2 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_02/img" 
	datafile_train2 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_02.hdf5"
	
	maskDir_test3 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_03/img" 
	datafile_test3 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_03.hdf5"
	maskDir_train3 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_03/img" 
	datafile_train3 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_03.hdf5"
	
	maskDir_test4 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_04/img" 
	datafile_test4 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_04.hdf5"
	maskDir_train4 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_04/img" 
	datafile_train4 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_04.hdf5"
	
	maskDir_test5 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_05/img" 
	datafile_test5 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/eval/box/txt/seq_05.hdf5"
	maskDir_train5 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_05/img" 
	datafile_train5 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_05.hdf5"
	
	maskDir_train6 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_06/img" 
	datafile_train6 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_06.hdf5"
	
	maskDir_train7 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_07/img" 
	datafile_train7 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_07.hdf5"
	
	maskDir_train8 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_08/img" 
	datafile_train8 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_08.hdf5"
	
	maskDir_train9 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_09/img" 
	datafile_train9 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_09.hdf5"
	
	maskDir_train10 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_10/img" 
	datafile_train10 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_10.hdf5"
	
	maskDir_train11 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_11/img" 
	datafile_train11 = "/content/drive/MyDrive/StereoSpikeA/datasets/EVIMO_Dataset/train/box/txt/seq_11.hdf5"
	
	
	general_config = "/content/drive/MyDrive/StereoSpike/general_config.yaml"

	genconfigs = yamlParams(general_config)
	crop = False
	incrementalPercent = 1
	
	print("data_motion")
	tsfm = transforms.Compose([
		RandomHorizontalFlip_ms(p=0.5),
		RandomVerticalFlip_ms(p=0.5),
		RandomEventDrop_ms(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
		])
		
	if(datasetType == "EVIMO"):
		database_train = EVIMODatasetBase(datafile_train, genconfigs, maskDir_train, crop, 
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_test = EVIMODatasetBase(datafile_test, genconfigs, maskDir_test, crop, 
							maxBackgroundRatio, incrementalPercent)
		database_train1 = EVIMODatasetBase(datafile_train1, genconfigs, maskDir_train1, crop, 
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_test1 = EVIMODatasetBase(datafile_test1, genconfigs, maskDir_test1, crop, 
							maxBackgroundRatio, incrementalPercent)
		database_train2 = EVIMODatasetBase(datafile_train2, genconfigs, maskDir_train2, crop, 
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_test2 = EVIMODatasetBase(datafile_test2, genconfigs, maskDir_test2, crop, 
							maxBackgroundRatio, incrementalPercent)
		database_train3 = EVIMODatasetBase(datafile_train3, genconfigs, maskDir_train3, crop, 
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_test3 = EVIMODatasetBase(datafile_test3, genconfigs, maskDir_test3, crop, 
							maxBackgroundRatio, incrementalPercent)
		database_train4 = EVIMODatasetBase(datafile_train4, genconfigs, maskDir_train4, crop, 
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_test4 = EVIMODatasetBase(datafile_test4, genconfigs, maskDir_test4, crop, 
							maxBackgroundRatio, incrementalPercent)
		database_train5 = EVIMODatasetBase(datafile_train5, genconfigs, maskDir_train5, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_test5 = EVIMODatasetBase(datafile_test5, genconfigs, maskDir_test5, crop,
							maxBackgroundRatio, incrementalPercent)
		database_train6 = EVIMODatasetBase(datafile_train6, genconfigs, maskDir_train6, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_train7 = EVIMODatasetBase(datafile_train7, genconfigs, maskDir_train7, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_train8 = EVIMODatasetBase(datafile_train8, genconfigs, maskDir_train8, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_train9 = EVIMODatasetBase(datafile_train9, genconfigs, maskDir_train9, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_train10 = EVIMODatasetBase(datafile_train10, genconfigs, maskDir_train10, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		database_train11 = EVIMODatasetBase(datafile_train11, genconfigs, maskDir_train11, crop,
							maxBackgroundRatio, incrementalPercent,tsfm)
		print("EVIMO used")
	else:
		raise Exception("Only EVIMO dataset with hdf5 format generated by preprocessing scripts handled with this code.")
		
	# uncomment if you want to split test/train using single hdf5 file
	torch.manual_seed(0)     
	#test_split = genconfigs['model']['testSplit'] # self.genconfigs['model']['testSplit']                      #

	train_dev_sets = torch.utils.data.ConcatDataset([database_train, database_train1, database_train2, database_train3, database_train4, database_train5, database_train6, database_train7, database_train8, database_train9, database_train10, database_train11])
	test_dev_sets = torch.utils.data.ConcatDataset([database_test, database_test1, database_test2, database_test3, database_test4, database_test5])

	# uncomment if you want to split test/train using single hdf5 file
	#num_workers = genconfigs['hardware']['readerThreads']
	#batch_size = genconfigs['batchsize']
			
	return train_dev_sets, test_dev_sets
		
def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    if not batch:
        return None

    return default_collate(batch)

def prepare_dataloader(dataset_train: Dataset, dataset_test: Dataset, batch_size: int, num_workers: int):
	train_loader = torch.utils.data.DataLoader(
			dataset_train,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=True,
			collate_fn=my_collate,
			drop_last=False,
			sampler=DistributedSampler(dataset_train))
			
	test_loader = torch.utils.data.DataLoader(
			dataset_test,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=True,
			collate_fn=my_collate,
			drop_last=False,
			sampler=DistributedSampler(dataset_test))
			
	return train_loader, test_loader
		
		
def load_train_objs(maxBackgroundRatio):
	lr = 5e-4
	wd = 1e-3
	train_set, test_set = dataSet(maxBackgroundRatio) # MyTrainDataset(2048)  # load your dataset
	model = PLIFNeuronPool_Separable_Pool3d_state(multiply_factor = 1.)  # load your model
	optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20, 40], gamma = 0.5)
	loss_module = BCEDiceLoss()
	return train_set, test_set, model, optimizer, scheduler, loss_module


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, num_workers: int, maxBackgroundRatio: float):
    ddp_setup(rank, world_size)
    dataset_train, dataset_test, model, optimizer, scheduler, loss_module = load_train_objs(maxBackgroundRatio)
    train_data, test_data = prepare_dataloader(dataset_train, dataset_test, batch_size, num_workers)
    print("Antes del Trainer")
    trainer = Trainer(model, train_data, test_data, optimizer, scheduler, rank, save_every, loss_module)
    print("Antes del train")
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, help='Workers')
    parser.add_argument('--maxBackgroundRatio', default=1.5, type=int, help='maxBackgroundRatio')
    
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.num_workers, args.maxBackgroundRatio), nprocs=world_size)

#https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
