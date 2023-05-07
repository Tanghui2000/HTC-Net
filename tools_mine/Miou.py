import torch
import numpy as np


def calculate_miou(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''

    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]]).cuda()  # 创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]]).cuda()  # 同上
    input = input.unsqueeze(1)  # 将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)  # 同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  # input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  # 同上
    batchMious = []  # 为该batch中每张图像存储一个miou
    mul = inputOht * targetOht  # 乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):  # 遍历图像
        ious = []
        for j in range(classNum):  # 遍历类别，包括背景
            intersection = torch.sum(mul[i][j])  # + 1e-6   #TP 1e-6用于出现mask全黑情况
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6 #FN+TP+FP
            if union == 1e-6:
                continue
            iou = intersection / union
            ious.append(iou.item())
        miou = np.mean(ious)  # 计算该图像的miou
        batchMious.append(miou)
    return np.mean(batchMious)


def calculate_mdice(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]]).cuda()  # 创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]]).cuda()  # 同上
    input = input.unsqueeze(1)  # 将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)  # 同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  # input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  # 同上
    batchMious = []  # 为该batch中每张图像存储一个miou
    mul = inputOht * targetOht  # 乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):  # 遍历图像
        dices = []
        for j in range(classNum):  # 遍历类别，包括背景
            intersection = 2 * torch.sum(mul[i][j]) + 1e-6  # 2TP
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6  #FN+2TP+FP
            dice = intersection / union
            dices.append(dice.item())
        Dice = np.mean(dices)  # 计算该图像的Dice
        batchMious.append(Dice)
    return np.mean(batchMious)


def Pa(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    tmp = input == target
    x = torch.sum(tmp).float()
    y = input.nelement()
    # print('x',x,y)
    return (x / y)


def pre(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP) / (TP + FP + 1e-6)
    return pre


def recall(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    recall = TP / (TP + FN + 1e-6)
    return recall


def F1score(input, target):
    input = input.data.cpu().numpy()
    target = target.data.cpu().numpy()
    # TP    predict 和 label 同时为1
    TP = ((input == 1) & (target == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((input == 0) & (target == 0)).sum()
    # FN    predict 0 label 1
    FN = ((input == 0) & (target == 1)).sum()
    # FP    predict 1 label 0
    FP = ((input == 1) & (target == 0)).sum()
    pre = (TP) / (TP + FP + 1e-6)
    recall = (TP) / (TP + FN + 1e-6)
    F1score = (2 * (pre) * (recall)) / (pre + recall + 1e-6)
    return F1score


def calculate_fwiou(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]]).cuda()  # 创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]]).cuda()  # 同上
    input = input.unsqueeze(1)  # 将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)  # 同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  # input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  # 同上
    batchFwious = []  # 为该batch中每张图像存储一个miou
    mul = inputOht * targetOht  # 乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):  # 遍历图像
        fwious = []
        for j in range(classNum):  # 遍历类别，包括背景
            TP_FN = torch.sum(targetOht[i][j])
            intersection = torch.sum(mul[i][j]) + 1e-6
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            if union == 1e-6:
                continue
            iou = intersection / union
            fwiou = (TP_FN / (input.shape[2] * input.shape[3])) * iou
            fwious.append(fwiou.item())
        fwiou = np.mean(fwious)  # 计算该图像的miou
        # print(miou)
        batchFwious.append(fwiou)
    return np.mean(batchFwious)
