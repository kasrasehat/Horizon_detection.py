import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
import scipy.io as sio
import torch
import argparse
from networks import CNNmodel1 as Net
#from Networks import FC_Net as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from torchviz import make_dot
import pickle
from canny_filter1 import edge_channel
import tqdm
# import my_transform

# main
# test data is extracted in this section
# in this section data normalized with min_max method

def train(args, model, device, train_loader, optimizer, epoch, x_train_label_dict):

    model.train()
    tot_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target.to(device)
        target[:,3] = (target[:,3] + 1)/2
        input = np.zeros((len(data), 5, 108, 192))
        
        for i in range(len(data)):
            path = x_train_label_dict[data[i].item()]
            frame = cv2.imread(path)
            grad, angle = edge_channel(frame)
            edge_features = np.append(np.reshape(grad, (108, 192, 1)),np.reshape(angle, (108, 192, 1)), axis = 2)
            frame = cv2.resize(frame, (192, 108))
            frame = np.append(frame, edge_features, axis = 2)
            frame = np.reshape(frame, (5, 108, 192))
            input[i, :, :, :] =  frame

        input = torch.Tensor(input/255).to(device)
        optimizer.zero_grad()
        output = model(input)
        loss_y   = F.mse_loss(output[0][:, 0], target[:,1], reduction='sum' )
        loss_y1 = F.mse_loss(output[0][:, 0], target[:, 1], reduction='mean')
        loss_sin = F.mse_loss(output[1], target[:, 2], reduction='sum')
        loss_cos = F.mse_loss(output[2], target[:, 3], reduction='sum')
        loss1 = loss_y + loss_sin +loss_cos
        ##loss.backward()
        loss1.backward()
        optimizer.step()
        tot_loss += loss1.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), tot_loss/((batch_idx+1) * len(data) )))

    print(' Epoch: {}\tLoss: {:.6f}'.format(epoch, tot_loss/(len(train_loader.dataset))))
    
    return output


def evaluation(args, model, device, valid_loader, optimizer, val_loss_min, epoch, x_valid_label_dict):
    model.eval()
    val_loss = 0
    val_loss1 = 0
    tot_loss = 0
    tot_y = 0
    tot_sin = 0
    tot_cos = 0

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data, target.to(device)
            target[:, 3] = (target[:, 3] + 1) / 2
            input = np.empty((0, 5, 108, 192), float)

            for i in range(len(data)):
                path = x_valid_label_dict[data[i].item()]
                frame = cv2.resize(cv2.imread(path), (192, 108))
                grad, angle = edge_channel(frame)
                edge_features = np.append(np.reshape(grad, (108, 192, 1)), np.reshape(angle, (108, 192, 1)), axis=2)
                frame = np.append(frame, edge_features, axis=2)
                frame = np.reshape(frame, (1, 5, 108, 192))
                input = np.append(input, frame, axis=0)

            input = torch.Tensor(input / 255).to(device)
            output = model(input)
            loss_y = F.mse_loss(output[0][:, 0], target[:, 1], reduction='sum')
            loss_sin = F.mse_loss(output[1], target[:, 2], reduction='sum')
            loss_cos = F.mse_loss(output[2], target[:, 3], reduction='sum')
            loss1 = loss_y + loss_sin + loss_cos
            tot_loss += loss1.item()
            tot_y += loss_y
            tot_sin += loss_sin
            tot_cos += loss_cos

    val_loss = tot_loss / (len(valid_loader.dataset))
    print('\nValidation average loss: {:.6f}\n'.format(val_loss))
    print('y average loss: {:.6f}'.format(tot_y / len(valid_loader.dataset)))
    print('sin average loss: {:.6f}'.format(tot_sin / len(valid_loader.dataset)))
    print('cos average loss: {:.6f}\n'.format(tot_cos / len(valid_loader.dataset)))

    # save model if validation loss has decreased
    if val_loss < val_loss_min and epoch == args.epochs:
        if args.save_model:

            #filename = 'model_epock_{0}_val_loss_{1}.pt'.format(epoch, val_loss)
            #torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            filename = 'weights/CNNmodel.pt'
            torch.save(model.state_dict(),filename)
            val_loss_min = val_loss
        return val_loss_min
    else:
        return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch finance EURUSD')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
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

   
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 4e-4)
    #weight_decay = 4e-4

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

    #transform = transforms.Compose([
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
    with open("E:/codes_py/horizon_detection/Data/data/x_train.txt", "rb") as fp:  # Pickling
        x_train = pickle.load(fp)
    with open("E:/codes_py/horizon_detection/Data/data/y_train.txt", "rb") as fp:  # Pickling
        y_train = pickle.load(fp)
    with open("E:/codes_py/horizon_detection/Data/data/x_valid.txt", "rb") as fp:  # Pickling
        x_valid = pickle.load(fp)
    with open("E:/codes_py/horizon_detection/Data/data/y_valid.txt", "rb") as fp:  # Pickling
        y_valid = pickle.load(fp)


    #convert list of strings to list of numbers
    # dictionary that maps integer to its string value 
    x_train_label_dict = {}

    # list to store integer labels 
    x_train_int_labels = []

    for i in range(len(x_train)):
        i = int(i)
        x_train_label_dict[i] = x_train[i]
        x_train_int_labels.append(i)

    tensor_x = torch.Tensor(x_train_int_labels)  # transform to torch tensor
    tensor_y = torch.Tensor(y_train)
    train_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset

    x_valid_label_dict = {}
    x_valid_int_labels = []

    for i in range(len(x_valid)):
        x_valid_label_dict[i] = x_valid[i]
        x_valid_int_labels.append(i)

    tensor_x = torch.Tensor(x_valid_int_labels)  # transform to torch tensor
    tensor_y = torch.Tensor(y_valid)
    valid_dataset = TensorDataset(tensor_x, tensor_y)

    train_loader = DataLoader(train_dataset, **kwargs_train)  # create your dataloader
    valid_loader = DataLoader(valid_dataset, **kwargs_val)


    val_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        output = train(args, model, device, train_loader, optimizer, epoch, x_train_label_dict)
        scheduler.step()
        out_loss = evaluation(args, model, device, valid_loader, optimizer, val_loss_min, epoch, x_valid_label_dict)
        if out_loss is not None:
            val_loss_min = out_loss

    #summary(model, args.batch_size)
    #make_dot(output.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)



if __name__ == '__main__':
    main()

