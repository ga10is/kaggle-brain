import os
import torch

IMG_SIZE = (256, 256)
INPUT_SIZE = (224, 224)
TEST_INPUT_SIZE = (224, 224)

INPUT = './data/'
INPUT_MNT = '/disk-brain/data'
TRAIN_PATH = os.path.join(INPUT, 'df_trn2.csv')
VALID_PATH = os.path.join(INPUT, 'df_val.csv')
TEST_PATH = os.path.join(INPUT, 'df_test.csv')
SUBMIT_PATH = os.path.join(INPUT, 'stage_1_sample_submission.csv')
TRAIN_IMG_PATH = os.path.join(INPUT, 'data2_512', 'stage_1_train_images_v2')
TEST_IMG_PATH = os.path.join(INPUT, 'data2_512', 'stage_1_test_images_v2')

USE_DCM = False

DEVICE = torch.device("cuda:0")

N_CLASSES = 2
N_SAMPLES = 100000
RUN_TTA = True
N_TTA = 4


VALID_RATIO = 0.10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
NUM_WORKERS = 8
PRINT_FREQ = 100
ITER_PER_CYCLE = 30
EPOCHS = ITER_PER_CYCLE * 4
# EPOCHS = 200

ADAM_LR = 5e-5
SGD_LR = 1e-2
MIN_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

FREEZE_EPOCH = 1
USE_PRETRAINED = False
RESET_OPT = False
PRETRAIN_PATH = './models/data224/seres50/402/best_model.pth'

DROPOUT_RATE = 0.2
LATENT_DIM = 512
TEMPERATURE = 60
MARGIN = 0.5

# visualize gradient flag
VIZ_GRAD = False
