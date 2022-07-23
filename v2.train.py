import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
import scipy.io as sio
import torch
import argparse
from networks import VGG_model as Net
# from Networks import FC_Net as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms,models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from torchviz import make_dot
import pickle
from canny_filter1 import edge_channel
import tqdm
from PIL import Image


def create_target(path):
    
    with open(path, 'r') as f:
        data = f.read()

    data = data.split(',')
    teta = np.arctan(-(int(data[3]) - int(data[1])) / 1920)*(2/np.pi)
    y_centr = int(data[5])/1080
    return np.array([y_centr, teta])


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    tot_loss = 0
    height = 224
    width = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        # transforms.RandomResizedCrop(224),
        #transforms.RandomCrop(224),
        # transforms.RandomGrayscale(p=0.4),
        # transforms.Grayscale(num_output_channels=3),
        # transforms.RandomAffine(45, shear=0.2),
        # transforms.ColorJitter(),
        # transforms.Lambda(utils.randomColor),
        # transforms.Lambda(utils.randomBlur),
        # transforms.Lambda(utils.randomGaussian),
        transforms.ToTensor(),
        normalize,
    ])
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data
        input_batch = np.empty((len(data), 3, height, width))
        target_batch = np.empty((len(data), 2))

        for i in range(len(data)):
            
            input_path = 'E:/HD/A/'+ str(int(data[i]))+'.jpg'
            target_path = 'E:/HD/A/'+ str(int(data[i]))+ '.xml'
            target = create_target(path = target_path)
            #frame = Image.open(input_path)
            frame = cv2.imread(input_path)
            #frame = cv2.resize(frame, (192, 108))
            #frame = np.reshape(frame, (3, 1080, 1920))
            frame = transform(frame)
            input_batch[i, :, :, :] = frame
            target_batch[i, :] = target

        inputs = torch.Tensor(input_batch).to(device)
        target = torch.tensor(target_batch, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss_y = F.mse_loss(output[:, 0], target[:, 0], reduction='sum')
        loss_y1 = F.mse_loss(output, target, reduction='mean')
        loss_teta = F.mse_loss(output[:,1], target[:, 1], reduction='sum')
        loss1 = loss_y + 10 * loss_teta
        #loss_y +
        ##loss.backward()
        loss1.backward()
        optimizer.step()
        tot_loss += loss1.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), tot_loss / ((batch_idx + 0.001) * len(data))))

    print(' Epoch: {}\tLoss: {:.6f}'.format(epoch, tot_loss / (len(train_loader.dataset))))

    return None


def evaluation(args, model, device, valid_loader, optimizer, val_loss_min, epoch):

    model.eval()
    height = 224
    width = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((height, width)),
        # transforms.RandomResizedCrop(224),
        # transforms.RandomCrop(224),
        # transforms.RandomGrayscale(p=0.4),
        # transforms.Grayscale(num_output_channels=3),
        # transforms.RandomAffine(45, shear=0.2),
        # transforms.ColorJitter(),
        # transforms.Lambda(utils.randomColor),
        # transforms.Lambda(utils.randomBlur),
        # transforms.Lambda(utils.randomGaussian),
        transforms.ToTensor(),
        normalize,
    ])
    val_loss = 0
    val_loss1 = 0
    tot_loss = 0
    tot_y = 0
    tot_teta = 0
    with torch.no_grad():
        for data, target in valid_loader:

            data = data
            input_batch = np.empty((len(data), 3, height, width))
            target_batch = np.empty((len(data), 2))

            for i in range(len(data)):
                input_path = 'E:/HD/A/' + str(int(data[i])) + '.jpg'
                target_path = 'E:/HD/A/' + str(int(data[i])) + '.xml'
                target = create_target(path=target_path)
                frame = cv2.imread(input_path)
                # frame = cv2.resize(frame, (192, 108))
                # frame = np.reshape(frame, (3, 1080, 1920))
                frame = transform(frame)
                input_batch[i, :, :, :] = frame
                target_batch[i, :] = target

            inputs = torch.Tensor(input_batch).to(device)
            target = torch.tensor(target_batch, dtype=torch.float32).to(device)
            output = model(inputs)
            loss_y = F.mse_loss(output[:, 0], target[:, 0], reduction='sum')
            loss_teta = F.mse_loss(output[:, 1], target[:, 1], reduction='sum')
            loss1 = loss_y + loss_teta
            tot_loss += loss1.item()
            tot_y += loss_y.item()
            tot_teta += loss_teta.item()


    val_loss = tot_loss / (len(valid_loader.dataset))
    print('\nValidation average loss: {:.6f}\n'.format(val_loss))
    print('y average loss: {:.6f}'.format(tot_y / len(valid_loader.dataset)))
    print('teta average loss: {:.6f}'.format(tot_teta / len(valid_loader.dataset)))

    # save model if validation loss has decreased
    if val_loss < val_loss_min and epoch == args.epochs:
        if args.save_model:
            # filename = 'model_epock_{0}_val_loss_{1}.pt'.format(epoch, val_loss)
            # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            filename = 'saved_models/CNNmodel1.pt'
            torch.save(model.state_dict(), filename)
            val_loss_min = val_loss
        return val_loss_min
    else:
        return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch horizon line detection')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.2, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--weight', default=False,
                        help='path of pretrain weights')
    parser.add_argument('--resume', default=False,
                        help='path of resume weights , "./cnn_83.pt" OR "./FC_83.pt" OR False ')

    args = parser.parse_args()
    use_cuda = args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)

    kwargs_train = {'batch_size': args.batch_size}
    kwargs_train.update({'num_workers': 1,
                         'shuffle': True,
                         'drop_last': True},
                        )
    kwargs_val = {'batch_size': args.valid_batch_size}
    kwargs_val.update({'num_workers': 1,
                       'shuffle': True})

    #model = models.ResNet
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=4e-4)
    # weight_decay = 4e-4

    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model.load_state_dict(checkpoint)

    # args.resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                model.load_state_dict(checkpoint)

    # transform = transforms.Compose([
    #    transforms.RandomHorizontalFlip(),
    #    transforms.RandomVerticalFlip(),
    #    transforms.RandomRotation(15),
    #    transforms.ToTensor(),
    #    normalize_dataset])

    ##################################
    # custom_transform = my_transform.Compose([
    #     transform.RandScale([args.scale_min, args.scale_max]),
    #     transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
    #     transform.RandomGaussianBlur(),
    #     transform.RandomHorizontalFlip(),
    #     transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
    #     transform.ToTensor(),
    #     transform.Normalize(mean=mean, std=std)])
    #####################################
    files = os.listdir('E:/HD/A/')
    num_of_files = int(len(files)/2)
    x = [int(i) for i in range(1, num_of_files)]
    y = [int(i) for i in range(1, num_of_files)]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.05, shuffle=True)

    tensor_x = torch.Tensor(x_train)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train)
    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

    tensor_x = torch.Tensor(x_valid)  # transform to torch tensor
    tensor_y = torch.Tensor(y_valid)
    valid_dataset = TensorDataset(tensor_x, tensor_y)

    train_loader = DataLoader(train_dataset, **kwargs_train)  # create your dataloader
    valid_loader = DataLoader(valid_dataset, **kwargs_val)

    val_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        output = train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        out_loss = evaluation(args, model, device, valid_loader, optimizer, val_loss_min, epoch)
        if out_loss is not None:
            val_loss_min = out_loss

    # summary(model, args.batch_size)
    # make_dot(output.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)


if __name__ == '__main__':
    main()

