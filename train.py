# Our Code is improved by Deng's work

import os

import torch
from torch.utils.tensorboard import SummaryWriter

from mvtec import get_data_transforms, MVTecDataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
from torch.utils.data import DataLoader
from encoder import wide_resnet50_2
from decoder import de_wide_resnet50_2
from layer_customization import connector

import torch.backends.cudnn as cudnn
from test import evaluation
from torch.nn import functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cal_anomaly_map(fs_list, ft_list, out_size=256, batch_size=8, device="cuda"):

    anomaly_map = torch.zeros([batch_size, out_size, out_size])
    anomaly_map = anomaly_map.to(device)

    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        a_map = 1 - F.cosine_similarity(fs, ft)

        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)

        a_map = a_map[:, 0, :, :]

        anomaly_map += a_map

    return anomaly_map

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function_dtl(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    loss += torch.mean(cos_loss(a.view(a.shape[0], -1), b.view(b.shape[0], -1)))
    return loss


def train(_class_):
    print(_class_)
    writer = SummaryWriter(f'./results/{_class_}')

    epochs = 300
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
    # momentum = 0.9
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')

    train_transform, test_transform, gt_transform = get_data_transforms(image_size)
    train_path = '/public/home/hpc214712266/yangyuxi/Datasets/MVTec-AD/' + _class_ + '/train'
    test_path = '/public/home/hpc214712266/yangyuxi/Datasets/MVTec-AD/' + _class_
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'


    train_data = ImageFolder(root=train_path, transform=train_transform)
    test_data = MVTecDataset(root=test_path, transform=test_transform, gt_transform=gt_transform, phase="test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder = wide_resnet50_2(pretrained=True)
    bn = connector()
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(bn.parameters())+list(decoder.parameters()), lr=learning_rate, betas=(0.9,0.999))

    mse_loss = torch.nn.MSELoss().to(device)

    for epoch in range(epochs):
        bn.train()
        decoder.train()

        loss_list = []
        loss_similarity_list = []
        for imgs, label in train_dataloader:

            normal, augment, gt_augment, gt_normal = imgs[0], imgs[1], imgs[2], imgs[3]

            normal = normal.to(device)
            augment = augment.to(device)
            gt_augment = gt_augment.to(device)
            gt_normal = gt_normal.to(device)

            gt_augment = gt_augment.to(torch.float32)
            gt_normal = gt_normal.to(torch.float32)

            inputs_normal = encoder(normal)
            inputs_augment = encoder(augment)

            i_normal = bn(inputs_normal)
            i_augment = bn(inputs_augment)

            outputs_normal = decoder(i_normal)
            outputs_augment = decoder(i_augment)

            anomaly_map_nor = cal_anomaly_map(inputs_normal, outputs_augment, augment.shape[-1], batch_size=augment.shape[0], device=device)
            anomaly_map_aug = cal_anomaly_map(inputs_augment, outputs_normal, normal.shape[-1], batch_size=normal.shape[0], device=device)

            anomaly_map_nor = anomaly_map_nor/3
            anomaly_map_aug = anomaly_map_aug/3

            loss_distillation = mse_loss(anomaly_map_aug, gt_augment) + mse_loss(anomaly_map_nor, gt_normal)

            loss_similarity = loss_function_dtl(i_normal, i_augment)

            optimizer.zero_grad()
            loss_distillation.backward()
            optimizer.step()

            loss_list.append(loss_distillation.item())
            loss_similarity_list.append(loss_similarity.item())


        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        writer.add_scalar('loss', np.mean(loss_list), epoch + 1)

        writer.add_scalar('loss_similarity', np.mean(loss_similarity_list), epoch + 1)


        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
            writer.add_scalar('auroc_sp', auroc_sp, epoch + 1)
            writer.add_scalar('auroc_px', auroc_px, epoch + 1)
            writer.add_scalar('aupro_px', aupro_px, epoch + 1)
            print('Sample Auroc{:.3f}, Pixel Auroc:{:.3f}, Pixel Aupro{:.3}'.format(auroc_sp, auroc_px, aupro_px))

            torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)

    return auroc_px, auroc_sp, aupro_px




if __name__ == '__main__':

    setup_seed(42)

    item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
                 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    for i in item_list:
        train(i)
