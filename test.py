import cv2
import torch
from networks import CNNmodel1 as Net
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle
from canny_filter import edge_channel
import numpy as np

if __name__ == "__main__":

    device = torch.device("cuda:0")
    model = Net().to(device)
    myload = torch.load("weights/CNNmodel.pt")
    try:
        model.load_state_dict(myload['state_dict'])
    except:
        model.load_state_dict(myload)


    with open("E:/codes_py/horizon_detection/Data/data/x_test.txt", "rb") as fp:  # Pickling
        x_test = pickle.load(fp)
    with open("E:/codes_py/horizon_detection/Data/data/y_test.txt", "rb") as fp:  # Pickling
        y_test = pickle.load(fp)

    kwargs_train = {'batch_size': 10}
    kwargs_train.update({'num_workers': 1,
                         'shuffle': False,
                         'drop_last': True})
    #convert list of strings to list of numbers
    # dictionary that maps integer to its string value 
    x_test_label_dict = {}

    # list to store integer labels 
    x_test_int_labels = []

    for i in range(len(x_test)):
        i = int(i)
        x_test_label_dict[i] = x_test[i]
        x_test_int_labels.append(i)

    tensor_x = torch.Tensor(x_test_int_labels)  # transform to torch tensor
    tensor_y = torch.Tensor(y_test)
    test_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    test_loader = DataLoader(test_dataset, **kwargs_train)  # create your dataloader

    model.eval()
    tot_loss = 0
    tot_y = 0
    tot_sin = 0
    tot_cos = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target.to(device)
            #target[:, 3] = (target[:, 3] + 1) / 2
            input = np.empty((0, 5, 108, 192), float)

            for i in range(len(data)):
                path = x_test_label_dict[data[i].item()]
                frame = cv2.resize(cv2.imread(path), (192, 108))
                grad, angle = edge_channel(frame)
                edge_features = np.append(np.reshape(grad, (108, 192, 1)), np.reshape(angle, (108, 192, 1)), axis=2)
                frame = np.append(frame, edge_features, axis=2)
                frame = np.reshape(frame, (1, 5, 108, 192))
                input = np.append(input, frame, axis=0)

            input = torch.Tensor(input / 255).to(device)
            output = model(input)
            output[2] = (output[2]* 2) - 1
            loss_y = F.mse_loss(output[0][:, 0], target[:, 1], reduction='sum')
            loss_sin = F.mse_loss(output[1], target[:, 2], reduction='sum')
            loss_cos = F.mse_loss(output[2], target[:, 3], reduction='sum')
            loss1 = loss_y + loss_sin + loss_cos
            tot_loss += loss1.item()
            tot_y += loss_y
            tot_sin += loss_sin
            tot_cos += loss_cos

    test_loss = tot_loss / (len(test_loader.dataset))
    print('\ntest average loss: {:.6f}\n'.format(test_loss))
    print('y average loss: {:.6f}'.format(tot_y / len(test_loader.dataset)))
    print('sin average loss: {:.6f}'.format(tot_sin / len(test_loader.dataset)))
    print('cos average loss: {:.6f}\n'.format(tot_cos / len(test_loader.dataset)))

        