from torchvision import datasets, transforms
import torch
from numpy import random
import numpy as np

def load_training(root_path, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         #transforms.RandomCrop(227),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         #transforms.Normalize(mean, std),
         ])
    data = datasets.ImageFolder(root=root_path , transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path,  batch_size,kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         #transforms.Resize([227, 227]),
         transforms.ToTensor(),
         ])
    data = datasets.ImageFolder(root=root_path , transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader


def crop_resize(data):
    #data_resize = []
    data_resize = np.zeros((0,3,227,227))
    #data:[32,3,255,255],output:[32,3,227,227]
    data = data.cpu().numpy()#numpy格式
    #print(data.shape)
    for i in range(32):
        x = random.randint(0, 29)
        y = random.randint(0, 29)
        data_i = data[i,:,:,:]
        data_i = np.expand_dims(data_i, axis=0)
        #print (data_i.shape)
        data_i = data_i[:,:,x:x+227,y:y+227]
        data_resize = np.vstack((data_resize, data_i))
    data_resize = np.array(data_resize)
    data_resize = torch.from_numpy(data_resize)
    data_resize = data_resize.float()
    #print (data_resize.shape)
    return data_resize


