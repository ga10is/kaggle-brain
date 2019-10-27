from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from apex import amp

from . import config
from .common.logger import get_logger
from .common.util import str_stats
from .helper_func import train_valid_split_v1
from .dataset import BrainDataset, BrainTTADataset, alb_trn_trnsfms, alb_val_trnsfms, alb_tst_trnsfms
from .model.model import HighSEResNeXt, HighCbamResNet, HighSEResNeXt2
from .model.metrics import AverageMeter, weighted_log_loss_metric
from .model.loss import WeightedBCE, FocalBCELoss
from .model.model_util import load_checkpoint, save_checkpoint, plot_grad_flow


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(epoch,
                    model,
                    loader,
                    criterion,
                    optimizer):
    loss_meter = AverageMeter()

    get_logger().info('[Start] epoch: %d' % epoch)
    get_logger().info('lr: %f' %
                      optimizer.state_dict()['param_groups'][0]['lr'])
    loader.dataset.update()

    if epoch < config.FREEZE_EPOCH:
        get_logger().info('freeze model parameter')
        # freeze pretrained layers
        for name, child in model.named_children():
            if name in ['feature']:
                for param in child.parameters():
                    param.requires_grad = False
    elif epoch == config.FREEZE_EPOCH:
        get_logger().info('unfreeze model parameter')
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True

    # train phase
    model.train()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img = img.to(config.DEVICE, dtype=torch.float)
        label = label.to(config.DEVICE, dtype=torch.float)

        with torch.set_grad_enabled(True):
            # mixed_img, label_a, label_b, lam = mixup_data(img, label, 1.0)

            logit = model(img)
            # print(logit.size())
            loss = criterion(logit, label)
            # loss = mixup_criterion(criterion, logit, label_a, label_b, lam)

            # backward
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            if config.VIZ_GRAD and i % config.PRINT_FREQ == 0:
                # visualize grad
                plot_grad_flow(model.named_parameters())

            optimizer.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item(), img.size(0))

        # print
        if i % config.PRINT_FREQ == 0:
            logit_cpu = logit.detach().cpu()
            get_logger().info('\n' + str_stats(logit_cpu[0].numpy()))
            prob = torch.sigmoid(logit_cpu)
            get_logger().info('\n' + str_stats(prob[0].numpy()))
            get_logger().info('train: %d loss: %f (just now)' % (i, loss_meter.val))
            get_logger().info('train: %d loss: %f' % (i, loss_meter.avg))

    get_logger().info("Epoch %d/%d train loss %f" %
                      (epoch, config.EPOCHS, loss_meter.avg))
    get_logger().info('GeM p: %f' % model.gem.p.item())
    return loss_meter.avg


def validate_one_epoch(epoch,
                       model,
                       loader,
                       criterion):
    loss_meter = AverageMeter()
    log_loss_meter = AverageMeter()

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img = img.to(config.DEVICE, dtype=torch.float)
        label = label.to(config.DEVICE, dtype=torch.float)
        with torch.no_grad():
            logit = model(img)

            loss = criterion(logit, label)
            loss_meter.update(loss.item(), img.size(0))

            prob_np = torch.sigmoid(logit.detach().cpu()).numpy()
            label_np = label.detach().cpu().numpy()
            log_loss = weighted_log_loss_metric(label_np, prob_np)
            log_loss_meter.update(log_loss, img.size(0))

        # print
        if i % config.PRINT_FREQ == 0:
            logit_cpu = logit.detach().cpu()
            get_logger().info('\n' + str_stats(logit_cpu[0].numpy()))
            prob = torch.sigmoid(logit_cpu)
            get_logger().info('\n' + str_stats(prob[0].numpy()))
            get_logger().info('vlaid: %d loss: %f metric: %f (just now)' %
                              (i, loss_meter.val, log_loss_meter.val))
            get_logger().info('valid: %d loss: %f metric: %f' %
                              (i, loss_meter.avg, log_loss_meter.avg))
    get_logger().info("Epoch %d/%d valid loss %f valid metric %f" %
                      (epoch, config.EPOCHS, loss_meter.avg, log_loss_meter.avg))

    return log_loss_meter.avg


def train():
    get_logger().info('Setting')
    get_logger().info('Image size: %s' % str(config.IMG_SIZE))

    # Load csv
    df_trn = pd.read_csv(config.TRAIN_PATH)
    df_val = pd.read_csv(config.VALID_PATH)

    df_trn = df_trn[df_trn['Image'] != 'ID_a47710c4b']

    # visualization train set
    print(df_trn.head())

    # visualization validate set
    print(df_val.head())

    # data information
    get_logger().info('train size: %d valid size: %d' % (len(df_trn), len(df_val)))
    get_logger().info('train positive ratio: %f valid positive ratio: %f' %
                      (df_trn['any'].mean(), df_val['any'].mean()))

    train_dataset = BrainDataset(
        df_trn, config.TRAIN_IMG_PATH, alb_trn_trnsfms, mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True
                              )
    valid_dataset = BrainDataset(
        df_val, config.TRAIN_IMG_PATH, alb_val_trnsfms, mode='valid')
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_TEST,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False
                              )

    model, criterion, optimizer, scheduler = init_model()
    start_epoch = 0
    if config.USE_PRETRAINED:
        get_logger().info('start with pretrained model: %s' % config.PRETRAIN_PATH)
        # load
        start_epoch, model, optimizer, scheduler, _ = load_checkpoint(
            model, optimizer, scheduler, config.PRETRAIN_PATH)
        if config.RESET_OPT:
            # reset optimizer
            start_epoch, optimizer, scheduler = reset_opt(model)

    get_logger().info('[Start] Training')
    best_score = 1e+8
    train_history = {'loss': []}
    valid_history = {'loss': []}
    for epoch in range(start_epoch + 1, config.EPOCHS + 1):

        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer)
        train_history['loss'].append(train_loss)

        if epoch % 1 == 0:
            valid_loss = validate_one_epoch(
                epoch, model, valid_loader, criterion)
            valid_history['loss'].append(valid_loss)

            is_best = valid_loss < best_score
            if is_best:
                best_score = valid_loss
            get_logger().info('best score (%f) at epoch (%d)' % (best_score, epoch))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best)
        # move scheduler.step to here
        scheduler.step()

    return train_history, valid_history


def init_model():
    torch.backends.cudnn.benchmark = True
    get_logger().info('Initializing classification model...')
    # model = HighResNet(dropout_rate=config.DROPOUT_RATE).to(config.DEVICE)
    model = HighSEResNeXt(dropout_rate=config.DROPOUT_RATE).to(config.DEVICE)
    # model = HighSEResNeXt2(dropout_rate=config.DROPOUT_RATE).to(config.DEVICE)
    # model = HighCbamResNet(dropout_rate=config.DROPOUT_RATE).to(config.DEVICE)

    # criterion = torch.nn.BCEWithLogitsLoss()
    label_weight = torch.tensor([1, 1, 1, 1, 1, 2]).to(
        config.DEVICE, dtype=torch.float)
    criterion = WeightedBCE(label_weight)
    # criterion = FocalBCELoss(
    #    bce_weight=0.7, label_weight=label_weight, gamma=2)
    # criterion = torch.nn.BCEWithLogitsLoss()
    '''
    optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                lr=config.SGD_LR,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.ITER_PER_CYCLE, config.MIN_LR)
    '''
    optimizer = optim.Adam([{'params': model.parameters()}], lr=config.ADAM_LR)
    mile_stones = [1, 2]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, mile_stones, gamma=0.5, last_epoch=-1)

    amp.register_float_function(torch, 'sigmoid')
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    return model, criterion, optimizer, scheduler


def reset_opt(model):
    get_logger().info('Change optimizer...')
    start_epoch = 0
    optimizer = optim.Adam([{'params': model.parameters()}], lr=config.ADAM_LR)
    mile_stones = [1, 2]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, mile_stones, gamma=0.5, last_epoch=-1)
    '''
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.SGD_LR,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.ITER_PER_CYCLE, config.MIN_LR)
    '''
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    return start_epoch, optimizer, scheduler


def predict():
    get_logger().info('Setting')

    # Load csv
    df_test = pd.read_csv(config.TEST_PATH)
    print(df_test.head())
    df_submit = pd.read_csv(config.SUBMIT_PATH)
    print(df_submit.head())

    get_logger().info('test size: %d ' % len(df_test))

    if config.RUN_TTA:
        # TTA
        test_dataset = BrainTTADataset(
            df_test, config.TEST_IMG_PATH, alb_trn_trnsfms, mode='predict', n_tta=config.N_TTA)
    else:
        # No TTA
        test_dataset = BrainDataset(
            df_test, config.TEST_IMG_PATH, alb_tst_trnsfms, mode='predict')

    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE_TEST,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False
                             )

    model, criterion, optimizer, scheduler = init_model()
    start_epoch = 0

    get_logger().info('predict with pretrained model: %s' % config.PRETRAIN_PATH)
    # load
    start_epoch, model, optimizer, scheduler, _ = load_checkpoint(
        model, optimizer, scheduler, config.PRETRAIN_PATH)
    get_logger().info('Predicting with loaded model(epoch: %d)' % start_epoch)

    get_logger().info('[Start] Predicting')
    preds = predict_labels(model, test_loader)

    df_submit = make_submit(df_submit, preds)
    print(df_submit.head())
    df_submit.to_csv('submission.csv', index=False)


def predict_labels(model, loader):
    """
    predict labels

    Returns
    -------
    nd.array, shape of (n_samples, 6)
    """
    preds = []

    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img, _ = data
        with torch.no_grad():
            if not config.RUN_TTA:
                img = img.to(config.DEVICE)
                logit = model(img)
                prob = torch.sigmoid(logit)
            else:
                # TTA: Dataset returns list of torch.Tensor
                img_list = img
                n_batches = img_list[0].size(0)
                prob = torch.zeros(
                    n_batches, 6, dtype=torch.float32).to(config.DEVICE)

                for img in img_list:
                    img = img.to(config.DEVICE)
                    logit = model(img)
                    prob += torch.sigmoid(logit)
                prob /= len(img_list)

        prob = prob.detach().cpu().numpy().reshape(-1)
        preds.append(prob)

    preds = np.concatenate(preds)
    return preds


def make_submit(df, preds):
    """
    make submission dataframe from submission sample and prediction
    """
    df.drop('Label', axis=1, inplace=True)
    df['Label'] = preds
    return df
