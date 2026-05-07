import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as F

from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate

from network.metrics import MeanDepthError, log_to_lin_depths, disparity_to_depth
from network.loss import Total_Loss

from datasets.MVSEC import load_MVSEC
from datasets.data_augmentation import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomTimeMirror, \
    RandomEventDrop

#from network.SNN_models import StereoSpike, fromZero_feedforward_multiscale_tempo_Matt_SpikeFlowNetLike
from network.SNN_models import StereoSpike, fromZero_feedforward_multiscale_tempo_monocular_SpikeFlowNetLike
from network.ANN_models import StereoSpike_equivalentANN

from network.metrics import MeanDepthError, log_to_lin_depths, disparity_to_depth
from network.loss import Total_Loss

from viz import show_learning, make_vid_from_pngs

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


######################
# GENERAL PARAMETERS #
######################

nfpdm = 1  # (!) don't choose it too big because of memory limitations (!)
N_warmup = 1
N_inference = 1
learned_metric = 'LIN'
show = True


###########################
# VISUALIZATION FUNCTIONS #
###########################

plt.ion()
fig = plt.figure()


########
# DATA #
########

# random transformations for data augmentation
tsfm = transforms.Compose([
    ToTensor(),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    # RandomTimeMirror(p=0.5),
    # RandomEventDrop(p=0.5, min_drop_rate=0., max_drop_rate=0.4)
])

test_set = load_MVSEC('./datasets/MVSEC/data/', scenario='indoor_flying', split='1',
                      num_frames_per_depth_map=nfpdm, warmup_chunks=1, train_chunks=1,
                      transform=tsfm, normalize=False, learn_on='LIN',
                      load_test_only=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=1,
                                               shuffle=False,
                                               drop_last=True,
                                               pin_memory=True)

