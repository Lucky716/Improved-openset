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


#########################针对二进制cross entropy loss的测试函数########################################################

#target_loss = -(t*torch.log(pred_softmax[:,10]) + (1-t)*torch.log(1.0 - pred_softmax[:,10]))

"""
import matplotlib.pyplot as plt
fig  = plt.figure()
#ax = fig.add_subplot(1,1,1)
x = np.arange(0,1,0.01)
x = list(x)
y1 = np.zeros(len(x))
y1  = list(y1)
y2 = np.zeros(len(x))
y2  = list(y2)
y3 = np.zeros(len(x))
y3  = list(y3)
y4 = np.zeros(len(x))
y4  = list(y4)
y5 = np.zeros(len(x))
y5  = list(y5)
y6 = np.zeros(len(x))
y6  = list(y6)
y7 = np.zeros(len(x))
y7  = list(y7)
y8 = np.zeros(len(x))
y8  = list(y8)
y9 = np.zeros(len(x))
y9  = list(y9)
"""
"""
t = np.arange(0.1,1,0.1)
y = np.zeros((len(t),len(x)))
for j in range(len(t)):
    for i in range(len(x)):
        y[j][i] = -(t[j] * np.log(x[i]) + (1 - t[j]) * np.log(1.0 - x[i]))
min = np.min(y)
print (min)
print (np.where(y==min))
min = np.where(y==min)
#index = y.index(min)
#print(x[index])
#plt.scatter(x[index], y[index])

plt.plot(x,y[0],color='green', label= 't=0.1')
plt.plot(x,y[1],color='red', label= 't=0.2')
plt.plot(x,y[2],color='pink', label= 't=0.3')
plt.plot(x,y[3],color='blue', label= 't=0.4')
plt.plot(x,y[4],color='black', label= 't=0.5')
plt.plot(x,y[5],color='yellowgreen', label= 't=0.6')
plt.plot(x,y[6],color='yellow', label= 't=0.7')
plt.plot(x,y[7],color='tomato', label= 't=0.8')
plt.plot(x,y[8],color='violet', label= 't=0.9')
#ax.annotate((x[index],y[index]),(x[index],y[index]))

"""
"""
for i in range(len(x)):
        y1[i] = -(0.1 * np.log(x[i]) + (1 - 0.1) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y2[i] = -(0.2 * np.log(x[i]) + (1 - 0.2) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y3[i] = -(0.3 * np.log(x[i]) + (1 - 0.3) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y4[i] = -(0.4 * np.log(x[i]) + (1 - 0.4) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y5[i] = -(0.5 * np.log(x[i]) + (1 - 0.5) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y6[i] = -(0.6 * np.log(x[i]) + (1 - 0.6) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y7[i] = -(0.7 * np.log(x[i]) + (1 - 0.7) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y8[i] = -(0.8 * np.log(x[i]) + (1 - 0.8) * np.log(1.0 - x[i]))
for i in range(len(x)):
        y9[i] = -(0.9 * np.log(x[i]) + (1 - 0.9) * np.log(1.0 - x[i]))

plt.plot(x,y1,color='yellowgreen', label= 't=0.1')
plt.plot(x,y2,color='blue', label= 't=0.2')
plt.plot(x,y3,color='tomato', label= 't=0.3')
plt.plot(x,y4,color='violet', label= 't=0.4')
plt.plot(x,y5,color='yellow', label= 't=0.5')
plt.plot(x,y6,color='green', label= 't=0.6')
plt.plot(x,y7,color='red', label= 't=0.7')
plt.plot(x,y8,color='black', label= 't=0.8')
plt.plot(x,y9,color='pink', label= 't=0.9')

#print (y1)
#print (type(y1))
min = np.min(y1)
#print (min)
index = y1.index(min)
#print (index)
if min ==y1[index]:
    plt.scatter(x[index], y1[index])

min = np.min(y2)
#print (min)
index = y2.index(min)
#print (index)
if min ==y2[index]:
    plt.scatter(x[index], y2[index])

min = np.min(y3)
#print (min)
index = y3.index(min)
#print (index)
if min ==y3[index]:
    plt.scatter(x[index], y3[index])

min = np.min(y4)
#print (min)
index = y4.index(min)
#print (index)
if min ==y4[index]:
    plt.scatter(x[index], y4[index])

min = np.min(y5)
#print (min)
index = y5.index(min)
#print (index)
if min ==y5[index]:
    plt.scatter(x[index], y5[index])


min = np.min(y6)
#print (min)
index = y6.index(min)
#print (index)
if min ==y6[index]:
    plt.scatter(x[index], y6[index])


min = np.min(y7)
#print (min)
index = y7.index(min)
#print (index)
if min ==y7[index]:
    plt.scatter(x[index], y7[index])


min = np.min(y8)
#print (min)
index = y8.index(min)
#print (index)
if min ==y8[index]:
    plt.scatter(x[index], y8[index])


min = np.min(y9)
#print (min)
index = y9.index(min)
#print (index)
if min ==y9[index]:
    plt.scatter(x[index], y9[index])

plt.legend() # 显示图例
plt.show()
"""
