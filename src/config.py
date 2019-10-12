import os
import torch

IMG_SIZE = (448, 448)
INPUT_SIZE = (382, 382)
TEST_INPUT_SIZE = (382, 382)  # (1024, 1024)

INPUT = './data/'
TRAIN_PATH = os.path.join(INPUT, 'df_trn.csv')
VALID_PATH = os.path.join(INPUT, 'df_val.csv')
TEST_PATH = os.path.join(INPUT, 'df_test.csv')
SUBMIT_PATH = os.path.join(INPUT, 'stage_1_sample_submission.csv')
TRAIN_IMG_PATH = os.path.join(INPUT, 'data_512', 'stage_1_train_images_jpg')
TEST_IMG_PATH = os.path.join(INPUT, 'data_512', 'stage_1_test_images_jpg')

DEVICE = torch.device("cuda:0")

N_CLASSES = 2
N_SAMPLES = 20000
RUN_TTA = False


VALID_RATIO = 0.10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 128
NUM_WORKERS = 8
PRINT_FREQ = 50
ITER_PER_CYCLE = 30
EPOCHS = ITER_PER_CYCLE * 4
# EPOCHS = 200

ADAM_LR = 5e-4
SGD_LR = 1e-2
MIN_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

FREEZE_EPOCH = 2
USE_PRETRAINED = False
RESET_OPT = False
PRETRAIN_PATH = './models/seres50/003/best_model.pth'

DROPOUT_RATE = 0.2
LATENT_DIM = 512
TEMPERATURE = 60
MARGIN = 0.5