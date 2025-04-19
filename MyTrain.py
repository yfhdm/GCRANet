import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import os

import pytorch_ssim
import pytorch_iou

from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from PIL import Image

from model.GCRANet import GCRANet
from sal_dataloader import SalObjDataset, RescaleT, RandomCrop, ToTensorLab
import  time
import cv2



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def train_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out +ssim_out
    return loss
def muti_loss(preds, label):
    loss = 0
    for i in range(0, len(preds)):
        loss = loss + train_loss(preds[i], label)
    return loss
def train(model_name, dataset_name):

    if dataset_name == "CrackSeg9k":
        epoch_num = 50
        epoch_val = 30
        bs = 8
    elif dataset_name == "DAGM":
        epoch_num = 200
        epoch_val = 150
        bs = 8
    elif dataset_name == "Magnetic-tile-defect-datasets":
        epoch_num = 900
        epoch_val = 700
        bs = 5
    elif dataset_name == "SD-saliency-900":
        epoch_num = 900
        epoch_val = 700
        bs = 8

    net = GCRANet()
    train_size = (256, 256)
    crop_size = (224, 224)

    file_dir= ".\\Datasets\\"
    train_image_root = os.path.join(file_dir, dataset_name + "\\train\images\\")
    train_gt_root = os.path.join(file_dir, dataset_name + "\\train\gt\\")
    test_image_root = os.path.join(file_dir, dataset_name + "\\test\images\\")
    test_gt_root = os.path.join(file_dir, dataset_name + "\\test\gt\\")

    train_img_list = [train_image_root + f for f in os.listdir(train_image_root)]
    train_gt_list = [train_gt_root + p for p in os.listdir(train_gt_root)]
    train_img_list = sorted(train_img_list)
    train_gt_list = sorted(train_gt_list)


    train_dataset = SalObjDataset(img_name_list=train_img_list,
                                      lbl_name_list=train_gt_list,
                                      is_edge=True,
                                      transform=transforms.Compose([
                                          RescaleT(train_size),
                                          RandomCrop(crop_size),
                                          ToTensorLab(flag=0)]
                                      ))

    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)

    # ------- 3. define model --------
    if torch.cuda.is_available():
        net = net.cuda()
    # ------- 4. define optimizer --------
    print("---define optimizer...")

    optimizer =  optimizer=optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- 5. training process --------
    print("---start training...")
    best_wf = 0
    for epoch in range(1, epoch_num + 1):
        print(epoch)
        start_time = time.time()

        for i, data in enumerate(train_dataloader):

            inputs, labels, edges = data['image'], data['label'], data['edge']


            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            edges = edges.type(torch.FloatTensor)
            #

            images, gts, edges = (Variable(inputs.to(device), requires_grad=False),
                                  Variable(labels.to(device), requires_grad=False),
                                  Variable(edges.to(device),requires_grad=False))

            optimizer.zero_grad()

            predictions_mask = net(images)
            mask_losses = 0
            for i in range(len(predictions_mask)):
                mask_losses = mask_losses + train_loss(predictions_mask[i], gts)

            mask_losses.backward()
            optimizer.step()


        end_time = time.time()
        print('Cost time: {:.4f}'.format(end_time - start_time))

        if epoch >= epoch_val:
            wf = eval_psnr(test_image_root, test_gt_root, train_size, net)
            if wf > best_wf:
                save_path = os.path.join("F:\\models\\", dataset_name)
                torch.save(net.state_dict(),
                           os.path.join(save_path, model_name + "-" + dataset_name + '-' + f'{wf:.4f}' + '.pth'))
                best_wf= wf


            net.train()


def eval_psnr(test_image_root, test_gt_root, train_size, model):
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    images = [test_image_root + f for f in os.listdir(test_image_root)]
    gts = [test_gt_root + p for p in os.listdir(test_gt_root)]
    images = sorted(images)
    gts = sorted(gts)

    test_salobj_dataset = SalObjDataset(img_name_list=images, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(train_size), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = inputs_test.cuda()

        gt = cv2.imread(gts[i_test], cv2.IMREAD_GRAYSCALE)

        H, W = gt.shape
        with torch.no_grad():
            predictions_mask = model(inputs_test)
            res = predictions_mask[0]

        res = res.data.cpu().numpy().squeeze()
        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert("L")
        pred = pred.resize((W, H), resample=Image.BILINEAR)

        pred = np.array(pred)
        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)



    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]



    return wfm
if __name__ == '__main__':



    model_name = "GCRANet"
    dataset_name = "Magnetic-tile-defect-datasets"
    train(model_name, dataset_name)







