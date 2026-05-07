import time
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate

from datasets.MVSEC import load_MVSEC
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomTimeMirror, \
    RandomEventDrop

from network.SNN_models import StereoSpike, fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike
from network.ANN_models import StereoSpike_equivalentANN

from network.metrics import MeanDepthError, log_to_lin_depths, disparity_to_depth
from network.loss import Total_Loss_Motion

from network import Params

from datasets.MVSEC import mvsec_dataset2 as evimo_base
from network.SNN_models2 import unetRNN6Layer_noBlock

from torch.utils.data.dataloader import default_collate

from viz import show_learning

import os

########################
# COMPLEMENT FUNCTIONS #
########################

def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    if not batch:
        return None

    return default_collate(batch)

##############################
# DEVICE AND REPRODUCIBILITY #
##############################

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def set_random_seed(seed):
    # Python
    random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if int(torch.__version__.split('.')[1]) < 8:
        torch.set_deterministic(True)  # for pytorch < 1.8
    else:
        torch.use_deterministic_algorithms(True)

    # NumPy
    np.random.seed(seed)


set_random_seed(2021)


######################
# GENERAL PARAMETERS #
######################

mode = 'depth'		     #depth/motion
datasetType = "EVIMO"

nfpdm = 1              # number of frames per depth map (1 label every 50 ms)
N_inference = 1        # number of chunks for training/testing (1 chunk = 50 ms = nfpdm frames)
N_warmup = 1           # number of chunks for warmup (if you want to use a stateful model)
batchsize = 1		
learned_metric = 'LIN' # learn metric depth ('LIN'), normalized log depth ('LOG') or disparity ('DISP')
learning_rate = 0.0002
weight_decay = 0.0
n_epochs = 70
show = False           # display network's predictions during training / validation


##############################################
# SPIKEMS PARAMETERS ONLY FOR MOTION SEGMENT #
##############################################

crop = False # True
maxBackgroundRatio = 1.5
checkpoint = "/content/drive/MyDrive/SpikeMS/pretrainedModels/EVIMO-pretrained/out/checkpoint.pth.tar" 
modeltype = "unetRNN6Layer_noBlock"
general_config = "/content/drive/MyDrive/StereoSpike/general_config.yaml"
maskDir = "/content/drive/MyDrive/datasets/EVIMO_Dataset/eval/table/txt/seq_02/img"
datafile = "/content/drive/MyDrive/datasets/EVIMO_Dataset/eval/table/txt/seq_02.hdf5"
incrementalPercent  = 1
saveImages = True
# saveInterval
# imageDir
imageLabel = ""
datasetType = "EVIMO"

output_dir = "/content/drive/MyDrive/SpikeMS/logs/" # os.path.join(os.getcwd(), 'logs')

genconfigs = yamlParams(general_config)
imageDir = "images" # Cambiar por una direccion
saveImageInterval = 1

###########################
# VISUALIZATION FUNCTIONS #
###########################

plt.ion()
fig = plt.figure()


########
# DATA #
########

if mode == 'depth':
    # random transformations for data augmentation
    tsfm = transforms.Compose([
    ToTensor(),
        # RandomHorizontalFlip(p=0.5),
        # RandomVerticalFlip(p=0.5),
        # RandomTimeMirror(p=0.5),
        # RandomEventDrop(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
    ])

    train_set, val_set, test_set = load_MVSEC('./datasets/MVSEC/data/', scenario='indoor_flying', split='1',
                                          num_frames_per_depth_map=nfpdm, warmup_chunks=1, train_chunks=1,
                                          transform=tsfm, normalize=False, learn_on='LIN')

    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                batch_size=batchsize,
                                                shuffle=True,
                                                drop_last=True,
                                                pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_set,
                                              batch_size=1,
                                              shuffle=False,
                                              drop_last=True,
                                              pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=True,
                                               pin_memory=True)

else:
    if(datasetType == "EVIMO"):
        database = base.EVIMODatasetBase(datafile, genconfigs, maskDir, crop, maxBackgroundRatio, incrementalPercent)
        print("EVIMO used")
    # elif(datasetType == "MOD"):
    #     database = base.MODDatasetBase(datafile, self.genconfigs, self.maskDir, crop, maxBackgroundRatio, incrementalPercent)
    #     print("MOD used")
    else:
        raise Exception("Only EVIMO dataset with hdf5 format generated by preprocessing scripts handled with this code.")
        # raise Exception("Only EVIMO or MOD datasets with hdf5 format generated by preprocessing scripts handled with this code.")
    
    dataset_size = len(database)
        # uncomment if you want to split test/train using single hdf5 file
        
    test_split = genconfigs['model']['testSplit'] # self.genconfigs['model']['testSplit']                      #
    test_size = int(test_split * dataset_size)                              #
    train_size = dataset_size - test_size                                   #

    torch.manual_seed(0)                                                    #
    train_dataset, test_dataset = torch.utils.data.random_split(database)    #

    # uncomment if you want to split test/train using single hdf5 file

    num_workers = genconfigs['hardware']['readerThreads']
    batch_size = genconfigs['batchsize']

    # self.loader = torch.utils.data.DataLoader(
    #         database,                                                       #
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=num_workers,
    #         pin_memory=True,
    #         collate_fn=my_collate,
    #         drop_last=False)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,                                                       #
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=my_collate,
            drop_last=False)        

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=my_collate,
            drop_last=False)        

    tb_writer = SummaryWriter(output_dir)    

###########
# NETWORK #
###########

if mode == 'depth':
    net = StereoSpike(surrogate_function=surrogate.ATan(), detach_reset=True, v_threshold=1.0, v_reset=0.).to(device)
    # net = StereoSpike_equivalentANN(activation_function=nn.Sigmoid()).to(device)
    # net = fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike(tau=3., v_threshold=1.0, v_reset=0.0, use_plif=True, multiply_factor=10.).to(device)
else:
    net = unetRNN6Layer_noBlock().to(device)


################
# OPTIMIZATION #
################

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 42, 60], gamma=0.5)
# loss_module = Total_Loss_Motion(simulation=1,alpha=0.5) #Init parameters

if mode == 'depth':
    loss_module = Total_Loss(alpha=0.5, scale_weights=(1., 1., 1., 1.), penalize_spikes=False)
else:
    loss_module = Total_Loss_Motion(simulation=1,alpha=0.5) #Init parameters


################
#    LOGGING   #
################

if mode == 'depth':
  logfile = open("./results/checkpoints/training_logs.txt", "w+")

  hyperparameters_report = \
      '''
      MODEL
      ------------------------------
      {}
      DATA
      ------------------------------
      nfpdm = {}
      N_train = {}
      training_set = {}
      test_set = {}
      learned_metric = {}
      data_augmentation = {}
      SHUFFLED TRAINING PROCEDURE
      ------------------------------
      batchsize = {}
      lr = {}
      wd = {}
      '''.format(net._get_name(),

                 nfpdm,
                 N_inference,
                 len(train_data_loader),
                 len(val_data_loader),
                 learned_metric,
                 tsfm,

                 batchsize,
                 learning_rate,
                 weight_decay)

logfile.write(hyperparameters_report)
print(hyperparameters_report)

tb_writer = SummaryWriter('./results/checkpoints/')


from torch._C import dtype
############
# TRAINING #
############

if mode == 'depth':
    for epoch in range(n_epochs):

        running_train_loss = 0
        running_train_MDE = 0
        running_test_loss = 0
        running_test_MDE = 0

        net.train()
        start_time = time.time()
        for init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, label in tqdm(
                train_data_loader):

            # Pass tensors on the GPU / CPU
            init_pots = init_pots.to(device)
            warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
            warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
            train_chunks_left = train_chunks_left.to(device, dtype=torch.float)
            train_chunks_right = train_chunks_right.to(device, dtype=torch.float)
            label = label.to(device)

            # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
            warmup_chunks_left = warmup_chunks_left.view(batchsize, N_warmup * nfpdm, 2, 260, 346)
            warmup_chunks_right = warmup_chunks_right.view(batchsize, N_warmup * nfpdm, 2, 260, 346)
            train_chunks_left = train_chunks_left.view(batchsize, N_inference * nfpdm, 2, 260, 346)
            train_chunks_right = train_chunks_right.view(batchsize, N_inference * nfpdm, 2, 260, 346)

            # concatenate subsequent frames channel-wise: (B, num_frames, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
            # where num_frames = num_chunks * nfpdm
            # Used to give some sort of temporal information to the stateless model via the input
            # /!\ number of filters in the first convolution should be changes accordingly /!\
            warmup_chunks_left = warmup_chunks_left.view(batchsize, 1, N_warmup * nfpdm * 2, 260, 346)
            warmup_chunks_right = warmup_chunks_right.view(batchsize, 1, N_warmup * nfpdm * 2, 260, 346)
            train_chunks_left = train_chunks_left.view(batchsize, 1, N_inference * nfpdm * 2, 260, 346)
            train_chunks_right = train_chunks_right.view(batchsize, 1, N_inference * nfpdm * 2, 260, 346)

            # concatenate left and right inputs channel-wise
            # (for binocular model, ignore for monocular model)
            warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
            train_chunks = torch.cat((train_chunks_left, train_chunks_right), dim=2)

            # initialize all neuron potentials
            functional.reset_net(net)

            # let intermediate neurons "warm up" and reach a steady state before "real" training
            # Useful for stateful models, but not used, as StereoSpike is stateless
            '''
            with torch.no_grad():
                net(warmup_chunks_left, warmup_chunks_right)
            '''

            # forward pass a long sequence of chunks
            pred, spks = net(train_chunks)  # for monocular models: pred, spks = net(test_chunks_left)

            # confront prediction and groundtruth
            if show:
                show_learning(fig, train_chunks_left, pred[0], label, 'train')

            # calculate loss and update weights with BPTT
            loss = loss_module(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            net.detach()

            # convert prediction and groundtruth back to linear (metric) depth, for Mean Depth Error (MDE) calculation
            # only consider full scale prediction for evaluation
            if learned_metric == 'LIN':
                lin_pred = pred[0]
                lin_label = label
            elif learned_metric == 'LOG':
                lin_pred = log_to_lin_depths(pred[0])
                lin_label = log_to_lin_depths(label)
            elif learned_metric == 'DISP':
                lin_pred = disparity_to_depth(pred[0])
                lin_label = disparity_to_depth(label)

            # calculate MDE
            MDE = MeanDepthError(lin_pred, lin_label)

            # save metrics
            running_train_loss += loss.item() * train_chunks_left.size(0)
            running_train_MDE += MDE

        # process saved metrics
        epoch_train_loss = running_train_loss / len(train_data_loader)
        epoch_train_MDE = running_train_MDE / len(train_data_loader)
        epoch_train_time = time.time() - start_time
        train_epoch_summary = "Epoch: {}, Training Loss: {}, Training Mean Depth Error (m): {}, Time: {}\n".format(epoch,
                                                                                                               epoch_train_loss,
                                                                                                               epoch_train_MDE,
                                                                                                               epoch_train_time)

        tb_writer.add_scalar('train_loss', epoch_train_loss, epoch)
        tb_writer.add_scalar('train_MDE', epoch_train_MDE, epoch)

        net.eval()
        with torch.no_grad():
            start_time = time.time()
            for init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label in tqdm(
                    val_data_loader):

                # Pass tensors on the GPU / CPU
                init_pots = init_pots.to(device)
                warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
                warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
                test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
                test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
                label = label.to(device)

                # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
                warmup_chunks_left = warmup_chunks_left.view(1, N_warmup * nfpdm, 2, 260, 346)
                warmup_chunks_right = warmup_chunks_right.view(1, N_warmup * nfpdm, 2, 260, 346)
                test_chunks_left = test_chunks_left.view(1, N_inference * nfpdm, 2, 260, 346)
                test_chunks_right = test_chunks_right.view(1, N_inference * nfpdm, 2, 260, 346)

                # concatenate train chunks channelwise: (B, num_frames, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
                # where num_frames = num_chunks * nfpdm
                # (for "tempo" feedforward ANN models, comment for other models)
                warmup_chunks_left = warmup_chunks_left.view(1, 1, N_warmup * nfpdm * 2, 260, 346)
                warmup_chunks_right = warmup_chunks_right.view(1, 1, N_warmup * nfpdm * 2, 260, 346)
                test_chunks_left = test_chunks_left.view(1, 1, N_inference * nfpdm * 2, 260, 346)
                test_chunks_right = test_chunks_right.view(1, 1, N_inference * nfpdm * 2, 260, 346)

                # concatenate left and right inputs channel-wise
                # (for binocular model)
                warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
                test_chunks = torch.cat((test_chunks_left, test_chunks_right), dim=2)

                functional.reset_net(net)

                '''
                net(warmup_chunks_left, warmup_chunks_right)
                '''

                pred, spks = net(test_chunks)  # for monocular models: pred, spks = net(test_chunks_left)

                if show:
                    show_learning(fig, test_chunks_left, lin_pred, label, 'eval')

                loss = loss_module(pred, label, spks)
                net.detach()

                if learned_metric == 'LIN':
                    lin_pred = pred[0]
                elif learned_metric == 'LOG':
                    lin_pred = log_to_lin_depths(pred[0])
                elif learned_metric == 'DISP':
                    lin_pred = disparity_to_depth(pred[0])

                MDE = MeanDepthError(lin_pred, label)

                running_test_loss += loss.item() / test_chunks_left.size(0)
                running_test_MDE += MDE

        epoch_test_loss = running_test_loss / len(val_data_loader)
        epoch_test_MDE = running_test_MDE / len(val_data_loader)
        epoch_test_time = time.time() - start_time
        test_epoch_summary = "Epoch: {}, Test Loss: {}, Test Mean Depth Error (m): {}, Time: {}\n".format(epoch,
                                                                                                      epoch_test_loss,
                                                                                                      epoch_test_MDE,
                                                                                                      epoch_test_time)
        print(train_epoch_summary + test_epoch_summary)
        logfile.write(train_epoch_summary + test_epoch_summary)

        tb_writer.add_scalar('test_loss', epoch_test_loss, epoch)
        tb_writer.add_scalar('test_MDE', epoch_test_MDE, epoch)

        # save model if better results
        if epoch_test_MDE < net.get_max_accuracy():
            print("Best performances so far: saving model...\n")
            logfile.write("Best performances so far: saving model...\n")
            torch.save(net.state_dict(), "./results/checkpoints/stereospike.pth")
            net.update_max_accuracy(epoch_test_MDE)

        net.increment_epoch()

        scheduler.step()

else:
    # self._trainfromscratch()    # train from scratch
    # self._loadNetFromCheckpoint(checkpoint, modeltype)
    # checkpoint = torch.load(checkpoint) # Creo que tampoco se puede porque está guardado específicamente con slayerpytorch
    # net.load_state_dict(checkpoint['state_dict']) # Aqui esto no va porque espera que se cargue con convoluciones tipo slayerpytorch
    net = net.eval()

    print("saving to: ", output_dir)
        
    total_MSE = 0
    total_IOU = 0
        
    scalar_i = 0
    tot_frames = 0

    if saveImages and not os.path.exists(imageDir):
                    os.mkdir(imageDir)

    for i, data in enumerate(tqdm(train_loader, desc='training')):  #loader --> train_loader
        if (data == None):
            continue

        # data = moveToGPUDevice(data, device, dtype)
              
        spikes_input = data['spike_tensor']
        spikes_masked = data['masked_spike_tensor']
        # spikes_input = spikes_input.ToTensor()

        print("\n")
        print("Original: ",spikes_input.shape)
        spikes_input = torch.sum(spikes_input, axis=0)
        spikes_masked = torch.sum(spikes_masked, axis=0)
        # spikes_input = np.squeeze(spikes_input, axis=0)
        # spikes_masked = np.squeeze(spikes_masked, axis=0)
        print("Squeeze: ", spikes_input.shape)
        spikes_input = spikes_input.permute(3,0,1,2)
        spikes_masked = spikes_masked.permute(3,0,1,2)
        print("Squeeze: ", spikes_input.shape)

        spikes_input = spikes_input.to(device)
        spikes_masked = spikes_masked.to(device)

        #model returns output and intermediate layers  
        try: 
            print("try")
            print("try: ", spikes_input.ndim)
            spike_pred =  net.forward(spikes_input).to(device)

        #model returns just the output layer
        except: 
            print("except")
            if spikes_input.ndim == 5:
                spikes_input = torch.sum(spikes_input, axis=0)
            print("except: ", spikes_input.ndim)
            spike_pred_arr = net.forward(spikes_input)
            spike_pred = spike_pred_arr[0].to(device)

        print("spike_pred_shape: ",spike_pred.shape)
        # print("spike_pred: ",spike_pred.size(3))
        if spike_pred.ndim == 3:
            spike_input_crop = spikes_input[ :, :spike_pred.size(1), :spike_pred.size(2)]
            spikes_masked_crop = spikes_masked[:, :spike_pred.size(1), :spike_pred.size(2)]
        else:
            spike_input_crop = spikes_input[:spike_pred.size(0), :, :spike_pred.size(2), :spike_pred.size(3)]
            spikes_masked_crop = spikes_masked[:spike_pred.size(0), :, :spike_pred.size(2), :spike_pred.size(3)]
        # spike_input_crop = spikes_input[:, :, :spike_pred.size(2), :spike_pred.size(3), :spike_pred.size(4)]
        # spikes_masked_crop = spikes_masked[:, :, :spike_pred.size(2), :spike_pred.size(3), :spike_pred.size(4)]

        spike_pred_2D = torch.sum(spike_pred, axis = (0,1))
        spike_mask_2D = torch.sum(spikes_masked_crop, axis = (0,1))
        # spike_mask_2D = torch.sum(spikes_masked_crop, axis = (0,1,2))
        print("shape spike_pred: ", spike_pred.ndim)
        print("shape spikes_masked_crop: ", spikes_masked_crop.ndim)
        print("shape spikes_masked_crop: ", spikes_masked_crop.shape)

        #calculate the loss and update the weights
        print("shape spike_pred_2d: ", spike_pred_2D.ndim)
        print("shape spike_mask_2D: ", spike_mask_2D.ndim)
        print("shape spike_pred_2d: ", spike_pred_2D.shape)
        print("shape spike_mask_2D: ", spike_mask_2D.shape)
        total_loss = loss_module(spike_pred_2D, spike_mask_2D)
        print("total_loss: ", total_loss)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        net.detach()

        #calculate metrics
        iou = getIOU(spike_pred_2D, spike_mask_2D)
        total_IOU += iou*len(data['ratio'])
        
        # ioucriterion = snn.loss(genconfigs).to(device) 
        # iou = ioucriterion.getIOU(spike_pred_2D, spike_mask_2D)
        # total_IOU += iou*len(data['ratio'])

        print(i, ": ", "curr iou", iou, data['ratio'])
        tb_writer.add_scalar('iou', iou.item(), scalar_i)

        scalar_i += 1
        tot_frames += len(data['ratio'])
        # tot_frames += len(ratio)

        # if saveImages and (i)%saveImageInterval== 0:
        #     spikes_inputnp = np.array(spikes_input.detach().cpu())
        #     spikes_maskednp = np.array(spikes_masked.detach().cpu())
        #     spikesPred_np = np.array(spike_pred.detach().cpu())

        #     print("save to: ", output_dir)

        #     for batch in range(0,spikes_input.shape[0]):
        #         print("largo de data[file_number]: ",len(data['file_number']))
        #         print("rango: ", len(range(0,spikes_input.shape[0])))
        #         print("batch: ", batch)
        #         curr_num = data['file_number'][batch]
        #         print("curr_num: ", curr_num)

        #         # print(type(spikesPred_np))
        #         # print(batch)
                
                
        #         # lenM = np.shape(spikesPred_np)
        #         # print(lenM)

        #         # print(np.shape(np.array(np.sum(spikesPred_np[batch,:,:,:], axis=(0)))))
                
        #         im = Image.fromarray(np.uint8(np.sum(spikesPred_np[batch,:,:,:], axis=(0))*255))
        #         im.save(os.path.join(imageDir,"_pred_epoch{}".format(curr_num) + imageLabel + ".jpg"))

        #         im2 = Image.fromarray(np.uint8(np.sum(spikes_maskednp[batch,:,:,:], axis=(0))*255))
        #         im2.save(os.path.join(imageDir,"_ideal_epoch{}".format(curr_num) + imageLabel + ".jpg"))

        #         # im3 = Image.fromarray(np.uint8(np.sum(spikes_inputnp[batch,:,:,:], axis=(0))*255))
        #         # im3.save(os.path.join(imageDir,"_input_epoch{}".format(curr_num)) + imageLabel + ".jpg")

    print("save to: ", output_dir)

    if saveImages:
        print("saving images to", os.getcwd(), imageDir)

    print("mean_IOU for {} batches of frames".format(tot_frames), total_IOU/tot_frames if tot_frames != 0 else 0)

print("training finished !")



# ############
# # TRAINING #
# ############

# for epoch in range(n_epochs):

#     running_train_loss = 0
#     running_train_MDE = 0
#     running_test_loss = 0
#     running_test_MDE = 0

#     net.train()
#     start_time = time.time()
#     for init_pots, warmup_chunks_left, warmup_chunks_right, train_chunks_left, train_chunks_right, label in tqdm(
#             train_data_loader):

#         # Pass tensors on the GPU / CPU
#         init_pots = init_pots.to(device)
#         warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
#         warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
#         train_chunks_left = train_chunks_left.to(device, dtype=torch.float)
#         train_chunks_right = train_chunks_right.to(device, dtype=torch.float)
#         label = label.to(device)

#         # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
#         warmup_chunks_left = warmup_chunks_left.view(batchsize, N_warmup * nfpdm, 2, 260, 346)
#         warmup_chunks_right = warmup_chunks_right.view(batchsize, N_warmup * nfpdm, 2, 260, 346)
#         train_chunks_left = train_chunks_left.view(batchsize, N_inference * nfpdm, 2, 260, 346)
#         train_chunks_right = train_chunks_right.view(batchsize, N_inference * nfpdm, 2, 260, 346)

#         # concatenate subsequent frames channel-wise: (B, num_frames, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
#         # where num_frames = num_chunks * nfpdm
#         # Used to give some sort of temporal information to the stateless model via the input
#         # /!\ number of filters in the first convolution should be changes accordingly /!\
#         warmup_chunks_left = warmup_chunks_left.view(batchsize, 1, N_warmup * nfpdm * 2, 260, 346)
#         warmup_chunks_right = warmup_chunks_right.view(batchsize, 1, N_warmup * nfpdm * 2, 260, 346)
#         train_chunks_left = train_chunks_left.view(batchsize, 1, N_inference * nfpdm * 2, 260, 346)
#         train_chunks_right = train_chunks_right.view(batchsize, 1, N_inference * nfpdm * 2, 260, 346)

#         # concatenate left and right inputs channel-wise
#         # (for binocular model, ignore for monocular model)
#         warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
#         train_chunks = torch.cat((train_chunks_left, train_chunks_right), dim=2)

#         # initialize all neuron potentials
#         functional.reset_net(net)

#         # let intermediate neurons "warm up" and reach a steady state before "real" training
#         # Useful for stateful models, but not used, as StereoSpike is stateless
#         '''
#         with torch.no_grad():
#             net(warmup_chunks_left, warmup_chunks_right)
#         '''

#         # forward pass a long sequence of chunks
#         pred, spks = net(train_chunks)  # for monocular models: pred, spks = net(test_chunks_left)

#         # confront prediction and groundtruth
#         if show:
#             show_learning(fig, train_chunks_left, pred[0], label, 'train')

#         # calculate loss and update weights with BPTT
#         loss = loss_module(pred, label)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         net.detach()

#         # convert prediction and groundtruth back to linear (metric) depth, for Mean Depth Error (MDE) calculation
#         # only consider full scale prediction for evaluation
#         if learned_metric == 'LIN':
#             lin_pred = pred[0]
#             lin_label = label
#         elif learned_metric == 'LOG':
#             lin_pred = log_to_lin_depths(pred[0])
#             lin_label = log_to_lin_depths(label)
#         elif learned_metric == 'DISP':
#             lin_pred = disparity_to_depth(pred[0])
#             lin_label = disparity_to_depth(label)

#         # calculate MDE
#         MDE = MeanDepthError(lin_pred, lin_label)

#         # save metrics
#         running_train_loss += loss.item() * train_chunks_left.size(0)
#         running_train_MDE += MDE

#     # process saved metrics
#     epoch_train_loss = running_train_loss / len(train_data_loader)
#     epoch_train_MDE = running_train_MDE / len(train_data_loader)
#     epoch_train_time = time.time() - start_time
#     train_epoch_summary = "Epoch: {}, Training Loss: {}, Training Mean Depth Error (m): {}, Time: {}\n".format(epoch,
#                                                                                                            epoch_train_loss,
#                                                                                                            epoch_train_MDE,
#                                                                                                            epoch_train_time)

#     tb_writer.add_scalar('train_loss', epoch_train_loss, epoch)
#     tb_writer.add_scalar('train_MDE', epoch_train_MDE, epoch)

#     net.eval()
#     with torch.no_grad():
#         start_time = time.time()
#         for init_pots, warmup_chunks_left, warmup_chunks_right, test_chunks_left, test_chunks_right, label in tqdm(
#                 val_data_loader):

#             # Pass tensors on the GPU / CPU
#             init_pots = init_pots.to(device)
#             warmup_chunks_left = warmup_chunks_left.to(device, dtype=torch.float)
#             warmup_chunks_right = warmup_chunks_right.to(device, dtype=torch.float)
#             test_chunks_left = test_chunks_left.to(device, dtype=torch.float)
#             test_chunks_right = test_chunks_right.to(device, dtype=torch.float)
#             label = label.to(device)

#             # reshape the inputs (B, num_chunks, nfpdm, 2, 260, 346) --> (B, num_chunks*nfpdm, 2, 260, 346)
#             warmup_chunks_left = warmup_chunks_left.view(1, N_warmup * nfpdm, 2, 260, 346)
#             warmup_chunks_right = warmup_chunks_right.view(1, N_warmup * nfpdm, 2, 260, 346)
#             test_chunks_left = test_chunks_left.view(1, N_inference * nfpdm, 2, 260, 346)
#             test_chunks_right = test_chunks_right.view(1, N_inference * nfpdm, 2, 260, 346)

#             # concatenate train chunks channelwise: (B, num_frames, 2, 260, 346) --> (B, 1, num_frames*2, 260, 346)
#             # where num_frames = num_chunks * nfpdm
#             # (for "tempo" feedforward ANN models, comment for other models)
#             warmup_chunks_left = warmup_chunks_left.view(1, 1, N_warmup * nfpdm * 2, 260, 346)
#             warmup_chunks_right = warmup_chunks_right.view(1, 1, N_warmup * nfpdm * 2, 260, 346)
#             test_chunks_left = test_chunks_left.view(1, 1, N_inference * nfpdm * 2, 260, 346)
#             test_chunks_right = test_chunks_right.view(1, 1, N_inference * nfpdm * 2, 260, 346)

#             # concatenate left and right inputs channel-wise
#             # (for binocular model)
#             warmup_chunks = torch.cat((warmup_chunks_left, warmup_chunks_right), dim=2)
#             test_chunks = torch.cat((test_chunks_left, test_chunks_right), dim=2)

#             functional.reset_net(net)

#             '''
#             net(warmup_chunks_left, warmup_chunks_right)
#             '''

#             pred, spks = net(test_chunks)  # for monocular models: pred, spks = net(test_chunks_left)

#             if show:
#                 show_learning(fig, test_chunks_left, lin_pred, label, 'eval')

#             loss = loss_module(pred, label, spks)
#             net.detach()

#             if learned_metric == 'LIN':
#                 lin_pred = pred[0]
#             elif learned_metric == 'LOG':
#                 lin_pred = log_to_lin_depths(pred[0])
#             elif learned_metric == 'DISP':
#                 lin_pred = disparity_to_depth(pred[0])

#             MDE = MeanDepthError(lin_pred, label)

#             running_test_loss += loss.item() / test_chunks_left.size(0)
#             running_test_MDE += MDE

#     epoch_test_loss = running_test_loss / len(val_data_loader)
#     epoch_test_MDE = running_test_MDE / len(val_data_loader)
#     epoch_test_time = time.time() - start_time
#     test_epoch_summary = "Epoch: {}, Test Loss: {}, Test Mean Depth Error (m): {}, Time: {}\n".format(epoch,
#                                                                                                   epoch_test_loss,
#                                                                                                   epoch_test_MDE,
#                                                                                                   epoch_test_time)
#     print(train_epoch_summary + test_epoch_summary)
#     logfile.write(train_epoch_summary + test_epoch_summary)

#     tb_writer.add_scalar('test_loss', epoch_test_loss, epoch)
#     tb_writer.add_scalar('test_MDE', epoch_test_MDE, epoch)

#     # save model if better results
#     if epoch_test_MDE < net.get_max_accuracy():
#         print("Best performances so far: saving model...\n")
#         logfile.write("Best performances so far: saving model...\n")
#         torch.save(net.state_dict(), "./results/checkpoints/stereospike.pth")
#         net.update_max_accuracy(epoch_test_MDE)

#     net.increment_epoch()

#     scheduler.step()

# print("training finished !")
