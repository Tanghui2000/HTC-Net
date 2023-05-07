import argparse
import os

import cv2
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from network.Net import model as ViT_seg
from network.Net import CONFIGS as CONFIGS_ViT_seg
from create_dataset_seg import Mydataset, test_transform
from tools_mine import Miou

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--imgs_test_path', '-ite', type=str,
                    default='dataset/ISIC2017/test_resize_224/Images', help='imgs val data path.')
parser.add_argument('--labels_test_path', '-lte', type=str,
                    default='dataset/ISIC2017/test_resize_224/Annotation', help='labels val data path.')
parser.add_argument('--csv_dir_test', '-cvt', type=str,
                    default='dataset/ISIC2017/csv/test.csv',
                    help='labels val data path.')

parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
parser.add_argument('--save_name', type=str, default='', help='checkpoint path')
parser.add_argument('--devicenum', default='0', type=str, help='use devicenum')
parser.add_argument('--device', default='cuda', type=str, help='use cuda')

parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
device = args.device
model_savedir = args.checkpoint + args.save_name + '/'
save_name = model_savedir + 'best'
test_transform = test_transform

df_test = pd.read_csv(os.path.join(os.getcwd(), args.csv_dir_test))
test_imgs, test_masks = args.imgs_test_path, args.labels_test_path

test_imgs = [''.join([test_imgs, '/', i]) for i in df_test['image_name']]
test_masks = [''.join([test_masks, '/', i.replace('.jpg', '_segmentation.png')]) for i in df_test['image_name']]
imgs_test = [cv2.imread(i)[:, :, ::-1] for i in test_imgs]
masks_test = [cv2.imread(i)[:, :, 0] for i in test_masks]

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

if __name__ == '__main__':
    test_number = len(test_imgs)
    test_ds = Mydataset(imgs_test, masks_test, test_transform)
    test_dl = DataLoader(test_ds, batch_size=1, pin_memory=False, num_workers=args.workers, )
    model.load_state_dict(torch.load(save_name + '.pth'))
    model.eval()
    test_mdice, test_miou, test_Pre, test_recall, test_F1score, test_pa = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            predicted = out.argmax(1)
            test_mdice += Miou.calculate_mdice(predicted, targets, 2).item()
            test_miou += Miou.calculate_miou(predicted, targets, 2).item()
            test_Pre += Miou.pre(predicted, targets).item()
            test_recall += Miou.recall(predicted, targets).item()
            test_F1score += Miou.F1score(predicted, targets).item()
            test_pa += Miou.Pa(predicted, targets).item()
    average_test_dice = test_mdice / test_number
    average_test_iou = test_miou / test_number
    average_test_Pre = test_Pre / test_number
    average_test_recall = test_recall / test_number
    average_test_F1score = test_F1score / test_number
    average_test_pa = test_pa / test_number

    f = open(model_savedir + 'log1' + '.txt', "a")
    f.write(
        '_dice' + str(float(average_test_dice)) + '  _miou' + str(average_test_iou) +
        '  _pre' + str(average_test_Pre) + '  _recall' + str(average_test_recall) +
        ' _f1_score' + str(average_test_F1score) + ' _pa' + str(average_test_pa) + '\n')
    f.close()
