import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp

def CrossMapLRN(size, alpha, beta, k=1.0, gpuDevice=0):
    lrn = SpatialCrossMapLRN_temp(size, alpha, beta, k, gpuDevice=gpuDevice)
    n = Lambda( lambda x,lrn=lrn: Variable(lrn.forward(x.data).cuda(gpuDevice)) if x.data.is_cuda else Variable(lrn.forward(x.data)) )
    return n


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

#torch.legacy.nn.

class AlexNet_DA(nn.Module):

    def __init__(self, num_classes=11, gpuDevice = 0):
        super(AlexNet_DA, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), (4, 4)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice),
            # Lambda(lambda x,lrn=SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data))),
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice),
            # Lambda(lambda x,lrn=SpatialCrossMapLRN(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            Lambda(lambda x: x.view(x.size(0), -1)),  # View,
            nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(9216, 4096)),
            # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(4096, 4096)),
            # Linear,
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,1000)),
            #nn.Softmax(),

        )

        self.generator = nn.Sequential(

            #nn.BatchNorm1d(1000),
            #nn.LeakyReLU(),
            #nn.Dropout(),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
        )

        self.label_classifier =nn.Sequential(

            nn.Linear(100, num_classes),
        )

    def forward(self,x):

        x = self.features(x)
        x = self.generator(x)
        x_l = self.label_classifier(x)
        return x_l



def load_pretrained_part(state_dict):
    own_state = state_dict()
    # print (own_state)
    for name, para in own_state.items():
        print (name)
        # print ("********************************************************")
    for name, param in state_dict.items():
        print (name)
        if name not in own_state:
            # print(name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

dict2 ={}

def AlexNet(pretrained=True,**kwargs):
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        #model.load_state_dict(torch.load("/home/fujiahui/vgg16-00b39a1b.pth"))
        pre_trained = torch.load("/home/fujiahui/caffe_to_torch_to_pytorch-master/AlexNet_torch_cpu.pth")
        dict2['features.0.weight'] = pre_trained['0.weight']
        dict2['features.0.bias'] = pre_trained['0.bias']
        dict2['features.4.weight'] = pre_trained['4.weight']
        dict2['features.4.bias'] = pre_trained['4.bias']
        dict2['features.8.weight'] = pre_trained['8.weight']
        dict2['features.8.bias'] = pre_trained['8.bias']
        dict2['features.10.weight'] = pre_trained['10.weight']
        dict2['features.10.bias'] = pre_trained['10.bias']
        dict2['features.12.weight'] = pre_trained['12.weight']
        dict2['features.12.bias'] = pre_trained['12.bias']
        dict2['features.16.1.weight'] = pre_trained['16.1.weight']
        dict2['features.16.1.bias'] = pre_trained['16.1.bias']
        dict2['features.19.1.weight'] = pre_trained['19.1.weight']
        dict2['features.19.1.bias'] = pre_trained['19.1.bias']
        dict2['features.22.1.weight'] = pre_trained['22.1.weight']
        dict2['features.22.1.bias'] = pre_trained['22.1.bias']

    model = AlexNet_DA(**kwargs)

    return model,dict2































