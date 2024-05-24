import os
import tqdm
import numpy as np
from torchsummary import summary

import torch
import torchvision
from torchvision import transforms

from nets.nn import resnet50
from utils.loss import yoloLoss
from utils.dataset import Dataset

import argparse
import re

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root = args.data_dir
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = resnet50()

    
    if(args.pre_weights != None):
        pattern = 'yolov1_([0-9]+)'
        strs = args.pre_weights.split('.')[-2]
        f_name = strs.split('/')[-1]
        epoch_str = re.search(pattern,f_name).group(1)
        epoch_start = int(epoch_str) + 1
        net.load_state_dict( \
            torch.load(f'./weights/{args.pre_weights}')['state_dict'])
    else:
        epoch_start = 1
        resnet = torchvision.models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
    
        net_dict = net.state_dict()
        for k in new_state_dict.keys():
            if k in net_dict.keys() and not k.startswith('fc'):
                net_dict[k] = new_state_dict[k]
        net.load_state_dict(net_dict)

    print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

    criterion = yoloLoss().to(device)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    #summary(net,input_size=(3,448,448))
    # different learning rate

    net.train()

    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr': learning_rate * 10}]
        else:
            params += [{'params': [value], 'lr': learning_rate}]

    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    with open('./Dataset/train.txt') as f:
        train_names = f.readlines()
    train_dataset = Dataset(root, train_names, train=True, transform=[transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=os.cpu_count())

    with open('./Dataset/test.txt') as f:
        test_names = f.readlines()
    test_dataset = Dataset(root, test_names, train=False, transform=[transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False,
                                            num_workers=os.cpu_count())

    print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
    print(f'BATCH SIZE: {batch_size}')

    for epoch in range(epoch_start,num_epochs):
        net.train()

        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # training
        total_loss = 0.
        print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
        progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            pred = net(images)
            
            optimizer.zero_grad()
            loss = criterion(pred, target.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch, num_epochs), total_loss / (i + 1), mem)
            progress_bar.set_description(s)
        
        
        # validation
        validation_loss = 0.0
        net.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(test_loader), total=len(test_loader))
            for i, (images, target) in progress_bar:
                images = images.to(device)
                target = target.to(device)

                prediction = net(images)
                loss = criterion(prediction, target)
                validation_loss += loss.data
            
        validation_loss /= len(test_loader)
        print(f'Validation_Loss:{validation_loss:07.3}')
        
        #if epoch % 5:
        #    save = {'state_dict': net.state_dict()}
        #    torch.save(save, f'./weights/yolov1_{epoch+1:04d}.pth')
        save = {'state_dict': net.state_dict()}
        torch.save(save, f'./weights/yolov1_{epoch:04d}.pth')

    save = {'state_dict': net.state_dict()}
    torch.save(save, './weights/yolov1_final.pth')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, default='./Dataset')
    parser.add_argument("--pre_weights", type=str, help="pretrained weight")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    
    args.pre_weights = 'yolov1_0010.pth'
    main(args)