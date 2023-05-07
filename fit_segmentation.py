import cv2
import os
import random
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tools_mine.Miou import Pa, calculate_miou


def write_options(model_savedir, args, best_acc_val):
    aaa = []
    aaa.append(['lr', str(args.lr)])
    aaa.append(['batch', args.batch_size])
    # aaa.append(['save_name', args.save_name])
    aaa.append(['seed', args.batch_size])
    aaa.append(['best_val_acc', str(best_acc_val)])
    aaa.append(['warm_epoch', args.warm_epoch])
    aaa.append(['end_epoch', args.end_epoch])
    f = open(model_savedir + 'option' + '.txt', "a")
    for option_things in aaa:
        f.write(str(option_things) + '\n')
    f.close()


def set_seed(seed=1):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    # 但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def fit(epoch, epochs, model, trainloader, valloader, device, criterion, optimizer, CosineLR):
    # with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
    scaler = GradScaler()
    if torch.cuda.is_available():
        model.to('cuda')
    running_loss = 0
    model.train()
    train_pa_whole = 0
    train_iou_whole = 0
    for batch_idx, (imgs, masks) in enumerate(trainloader):
        # t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
        imgs, masks_cuda = imgs.to(device), masks.to(device)
        imgs = imgs.float()
        with autocast():
            masks_pred = model(imgs)
            loss = criterion(masks_pred, masks_cuda)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # with torch.no_grad():
        predicted = masks_pred.argmax(1)
        # train_pa = Pa(predicted,masks_cuda)
        train_iou = calculate_miou(predicted, masks_cuda, 2)
        # train_pa_whole += train_pa.item()
        train_iou_whole += train_iou.item()
        running_loss += loss.item()
        # epoch_acc = train_pa_whole/(batch_idx+1)
        epoch_iou = train_iou_whole / (batch_idx + 1)
        # t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
        #               train_pa='{:.2f}%'.format(epoch_acc*100),train_iou='{:.2f}%'.format(epoch_iou*100))
        # t.update(1)
    # epoch_acc = correct / total
    epoch_loss = running_loss / len(trainloader.dataset)
    # with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
    val_running_loss = 0
    val_pa_whole = 0
    val_iou_whole = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(valloader):
            # t.set_description("val(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            masks_pred = model(imgs)
            predicted = masks_pred.argmax(1)
            # val_pa = Pa(predicted,masks_cuda)
            val_iou = calculate_miou(predicted, masks_cuda, 2)
            # val_pa_whole += val_pa.item()
            val_iou_whole += val_iou.item()
            loss = criterion(masks_pred, masks_cuda)
            val_running_loss += loss.item()
            epoch_val_acc = val_pa_whole / (batch_idx + 1)
            epoch_val_iou = val_iou_whole / (batch_idx + 1)
            # t.set_postfix(loss='{:.3f}'.format(val_running_loss / (batch_idx + 1)),
            #               val_pa='{:.2f}%'.format(epoch_val_acc*100),val_iou='{:.2f}%'.format(epoch_val_iou*100))
            # t.update(1)
        # epoch_test_acc = test_correct / test_total
    epoch_val_loss = val_running_loss / len(valloader.dataset)
    # if epoch > 2:
    CosineLR.step()
    # if epoch > 2:

    return epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou