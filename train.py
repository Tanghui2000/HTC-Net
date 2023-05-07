import cv2
import os
import torch
import copy
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from fit_segmentation import set_seed
import pandas as pd
from tqdm import tqdm
from create_dataset_seg import Mydataset, for_train_transform1, test_transform
from fit_segmentation import fit
import argparse
from network.Net import model as ViT_seg
from network.Net import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_train_path', '-it', type=str,
                    default='dataset/ISIC2017/train_resize_224/Images', help='imgs train data path.')
parser.add_argument('--labels_train_path', '-lt', type=str,
                    default='dataset/ISIC2017/train_resize_224/Annotation', help='labels train data path.')
parser.add_argument('--imgs_val_path', '-iv', type=str,
                    default='dataset/ISIC2017/val_resize_224/Images', help='imgs val data path.')
parser.add_argument('--labels_val_path', '-lv', type=str,
                    default='dataset/ISIC2017/val_resize_224/Annotation', help='labels val data path.')
parser.add_argument('--csv_dir_train', '-ct', type=str,
                    default='dataset/ISIC2017/csv/train.csv',
                    help='labels val data path.')
parser.add_argument('--csv_dir_val', '-cv', type=str,
                    default='dataset/ISIC2017/csv/val.csv', help='labels val data path.')

parser.add_argument('--resize', default=224, type=int, help='resize shape')
parser.add_argument('--batch_size', default=8, type=int, help='batchsize')
parser.add_argument('--workers', default=16, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--warm_epoch', '-w', default=0, type=int, help='end epoch')
parser.add_argument('--end_epoch', '-e', default=50, type=int, help='end epoch')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')
parser.add_argument('--tb_path', type=str, default='log/', help='tensorboard path')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default='', help='checkpoint path')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')

parser.add_argument('--seed', default='0', type=int, help='seed_num')
parser.add_argument('--train_number', default='400', type=int, help='seed_num')

parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one cfg model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum

begin_time = time.time()
set_seed(seed=2021)
device = args.device
model_savedir = args.checkpoint + args.save_name + '/'
save_name = model_savedir + 'best'

print(model_savedir)
if not os.path.exists(model_savedir):
    os.mkdir(model_savedir)
if not os.path.exists(save_name):
    os.mkdir(save_name)
epochs = args.warm_epoch + args.end_epoch
print('Start loading data.')
res = args.resize
df_train = pd.read_csv(os.path.join(os.getcwd(), args.csv_dir_train))  # [:args.train_number]
df_val = pd.read_csv(os.path.join(os.getcwd(), args.csv_dir_val))   #getcwd返回当前进程的工作目录
df_test = pd.read_csv(os.path.join(os.getcwd(), args.csv_dir_test))
train_imgs, train_masks = args.imgs_train_path, args.labels_train_path
val_imgs, val_masks = args.imgs_val_path, args.labels_val_path
test_imgs, test_masks = args.imgs_test_path, args.labels_test_path

train_imgs = [''.join([train_imgs, '/', i]) for i in df_train['image_name']]
train_masks = [''.join([train_masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in df_train['image_name']]
val_imgs = [''.join([val_imgs, '/', i]) for i in df_val['image_name']]
val_masks = [''.join([val_masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in df_val['image_name']]

imgs_train = [cv2.imread(i)[:, :, ::-1] for i in train_imgs]
masks_train = [cv2.imread(i)[:, :, 0] for i in train_masks]
imgs_val = [cv2.imread(i)[:, :, ::-1] for i in val_imgs]
masks_val = [cv2.imread(i)[:, :, 0] for i in val_masks]
train_transform = for_train_transform1()
test_transform = test_transform

after_read_date = time.time()
print('data_time', after_read_date - begin_time)

best_acc_final = []


def main():
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    model.load_from()
    train_ds = Mydataset(imgs_train, masks_train, train_transform)
    val_ds = Mydataset(imgs_val, masks_val, test_transform)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
                          drop_last=True, prefetch_factor=4)

    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=False, num_workers=args.workers,
                        prefetch_factor=4)
    best_iou = 0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    with tqdm(total=epochs, ncols=60, ascii=True) as t:
        for epoch in range(epochs):
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch, epochs, model, train_dl, val_dl, device, criterion, optimizer, CosineLR)

            f = open(model_savedir + 'log' + '.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss' + str(epoch_loss) + '  _val_loss' + str(epoch_val_loss) +
                    ' _train_miou' + str(epoch_iou) + ' _val_iou' + str(epoch_val_iou) + '\n')
            if epoch_val_iou > best_iou:
                f.write('\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_iou = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name, '.pth']))
            f.close()

            train_loss.append(epoch_loss)
            train_acc.append(epoch_iou)
            val_loss.append(epoch_val_loss)
            val_acc.append(epoch_val_iou)
            t.update(1)
    print('trained successfully. Best AP:{:5f}'.format(best_iou))


if __name__ == '__main__':
    main()
after_net_time = time.time()
print('net_time', after_net_time - after_read_date)
print('best_acc_final', best_acc_final)
print(save_name, '\n', 'finish')
