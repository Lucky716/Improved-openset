from __future__ import print_function
import argparse
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import AlexNet as models
from PIL import Image
import numpy as np
from torch.utils import model_zoo
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Training settings
batch_size = 32
epochs = 500
lr = 0.001
momentum = 0.9
no_cuda =False
seed = 8
log_interval = 10
num_step = 1
l2_decay = 5e-4
#root_path = "./dataset/"
##very important
averge = np.load("/home/fujiahui/Desktop/openset_caffe/ilsvrc_2012_mean.npy")
averge = torch.from_numpy(averge)
averge = averge.float()

###适用于D-A，W-A，D-W，W-D

source_name = "D"
target_name = "A"
source_path = "/home/fujiahui/Desktop/openset_caffe/office-31/D_source/images"
target_path_train = "/home/fujiahui/Desktop/openset_caffe/office-31/A_target/images"  #分一下训练集和测试集
target_path_test = "/home/fujiahui/Desktop/openset_caffe/office-31/A_target/images"


cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(source_path, batch_size ,kwargs)

target_train_loader = data_loader.load_training(target_path_train, batch_size, kwargs)
target_test_loader = data_loader.load_testing(target_path_test, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
print (len_source_dataset)
print (len_target_dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)
print (len_source_loader)
print (len_target_loader)
#引入预训练模型


def load_pretrain(model,pre_trained):
    model_dict = model.state_dict()

    for k, v in model_dict.items():
        print(k)
        #if not "generator.0.weight" in k and not "generator.0.bias" in k and not "generator.1.weight" in k and not "generator.1.bias" in k and not "generator.1.running_mean" in k and not "generator.1.running_var" in k and not "label_classifier.0.weight" in k and not "label_classifier.0.bias"in k and not "label_classifier.1.weight" in k and not "label_classifier.1.bias"in k and not "label_classifier.1.running_mean" in k and not "label_classifier.1.running_var" in k and not "label_classifier.4.weight" in k and not "label_classifier.4.bias"in k and not "label_classifier.5.weight" in k and not "label_classifier.5.bias"in k and not "label_classifier.5.running_mean" in k and not "label_classifier.5.running_var" in k and not "label_classifier.8.weight" in k and not "label_classifier.8.bias"in k :
        if not "generator.0.weight" in k and not "generator.0.bias" in k  and not "generator.1.weight" in k and not "generator.1.bias" in k and not "generator.1.running_mean" in k and not "generator.1.running_var" in k and not "label_classifier.0.weight" in k and not "label_classifier.0.bias" in k :
            #model_dict[k] = pre_trained[k[k.find(".") + 1:]]
            #print (k[k.find(".") + 1:])
            model_dict[k] = pre_trained[k]

    model.load_state_dict(model_dict)
    return model


def train(epoch,model):
    #最后的全连接层学习率为前面的10倍
    #LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    LEARNING_RATE = lr
    print("learning rate:", LEARNING_RATE)
    optimizer_fea = torch.optim.SGD([
        {'params': model.generator.parameters()},
    ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    optimizer_critic = torch.optim.SGD([
        {'params': model.label_classifier.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)


    #loss的计算
    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)
    #dlabel_src = Variable(torch.ones(batch_size).long().cuda())
    #dlabel_tgt = Variable(torch.zeros(batch_size).long().cuda())
    i = 1
    while i <= len_target_loader:
        model.train()

        target_data, target_label = data_target_iter.next()
        target_data = target_data * 255
        target_data = target_data - averge
        # crop_resize
        target_data = data_loader.crop_resize(target_data)
        if cuda:
            target_data, target_label = target_data.cuda(), target_label.cuda()
        target_data = Variable(target_data)
        pred_target = model(target_data)
        t = 1
        # print(pred_target.data.cpu().numpy().shape)#32*11
        pred_softmax = F.softmax(pred_target)  # 32*11
        target_loss = -(t*torch.log(pred_softmax[:,10]) + (1-t)*torch.log(1.0 - pred_softmax[:,10]))
        target_loss = torch.mean(target_loss)
        """
        p = pred_softmax[:, 10]

        target_loss = (0.5 * (t * (math.log(t + 1e-8) - torch.log(p + 1e-8)) + (1 - t) * ( math.log(1 - t + 1e-8) - torch.log(1 - p + 1e-8))
                    + p * (torch.log(p + 1e-8) - math.log(t + 1e-8)) + (1 - p) * (torch.log(1 - p + 1e-8) - math.log(1 - t + 1e-8)))).mean()

        ####target_loss = (t * (math.log(t + 1e-8) - torch.log(p + 1e-8)) + (1 - t) * (math.log(1 - t + 1e-8) - torch.log(1 - p + 1e-8))).mean()
        """


        source_data, source_label = data_source_iter.next()
        #print (source_data.cpu().numpy().shape)
        source_data = source_data*255
        source_data = source_data - averge
        #crop_resize
        source_data = data_loader.crop_resize(source_data)
        if i % len_source_loader == 0:
            data_source_iter = iter(source_loader)
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        pred_source = model(source_data)
        label_loss = F.nll_loss(F.log_softmax(pred_source), source_label)
        #本质上nll_loss和cross_entropy得到的结果是一样的，只是cross_entropy帮你做softmax的过程


        #二进制的cross entripy的函数表示：binary_cross_entropy




        loss1 = label_loss - target_loss####G
        loss2 = label_loss + target_loss####C

        optimizer_fea.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer_fea.step()
        optimizer_fea.zero_grad()
        optimizer_critic.zero_grad()
        loss2.backward(retain_graph=True)
        optimizer_critic.step()

        if i % log_interval == 0:
            print ('num_iter:',i,'label_loss:',label_loss.data[0],'target_loss:',target_loss.data[0],'loss_G:',loss1.data[0],'loss_C:',loss2.data[0])

        i = i + 1

def test(epoch ,model):
    model.eval()
    test_loss = 0
    correct_num = 0
    Dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    total_num_Dict = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    for data, target in target_test_loader:
        data = data*255
        data = data - averge
        data = data[:,:,14:241,14:241]#resize
        #print (type(target))
        #print (target.shape)
        #print (target)
        #print ('******')

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output), target, size_average=False).data[0] # sum up batch loss
        pred = s_output.data.max(1)[1] # get the index of the max log-probability
        correct_num += pred.eq(target.data.view_as(pred)).cpu().sum()

        target = target.data.cpu()
        for i in range(target.numpy().shape[0]):
            total_num_Dict[str(target[i])] += 1
            if target[i] == pred[i]:
                Dict[str(pred[i])] += 1
            else:
                pass
    print(Dict)
    print(total_num_Dict)
    acc0 = Dict['0'] / total_num_Dict['0']
    acc1 = Dict['1'] / total_num_Dict['1']
    acc2 = Dict['2'] / total_num_Dict['2']
    acc3 = Dict['3'] / total_num_Dict['3']
    acc4 = Dict['4'] / total_num_Dict['4']
    acc5 = Dict['5'] / total_num_Dict['5']
    acc6 = Dict['6'] / total_num_Dict['6']
    acc7 = Dict['7'] / total_num_Dict['7']
    acc8 = Dict['8'] / total_num_Dict['8']
    acc9 = Dict['9'] / total_num_Dict['9']
    acc10 = Dict['10'] / total_num_Dict['10']
    correct_all = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10) / 11 * 100.
    correct_part = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9) / 10 * 100.
    correct = 100. * correct_num / len_target_dataset
    test_loss /= len_target_dataset

    print('\n{} set: Average loss: {:.4f}, True accuracy: {}/{} ({:.2f}%) All acc({:.2f}%) Part acc({:.2f}%)\n'.format(
        target_name, test_loss, correct_num, len_target_dataset, correct, correct_all, correct_part))
    return correct, correct_all, correct_part, acc10 * 100.



if __name__ == '__main__':
    #model = models.AlexNet_DA()
    model ,pre_trained = models.AlexNet(True)
    correct = 0
    outcome_all_init = 0
    outcome_part_init = 0
    correct1 = []
    correct2 = []
    correct3 = []
    acc10 = []
    print(model)
    if cuda:
        model.cuda()
    model = load_pretrain(model,pre_trained)

    for epoch in range(1, epochs + 1):
        print ('epoch:',epoch)
        train(epoch,model)
        t_correct,outcome_all,outcome_part, acc = test(epoch ,model)
        correct1.append(t_correct)
        correct2.append(outcome_all)
        correct3.append(outcome_part)
        acc10.append(acc)
        if t_correct > correct:
            correct = t_correct
        if outcome_all > outcome_all_init:
            outcome_all_init = outcome_all
        if outcome_part > outcome_part_init:
            outcome_part_init = outcome_part
        print('source: {} to target: {}  max accuracy{: .2f}% max average all accuracy{: .2f}%  max average part accuracy{: .2f}%\n'.format(
              source_name, target_name, correct ,outcome_all_init,outcome_part_init))
    x = np.arange(0, 500, 1)
    x = list(x)

    fig = plt.figure()
    plt.plot(x, correct1, color='green', label='D-A True acc')
    plt.plot(x, correct2 , color='red', label='D-A OS(11)')
    plt.plot(x, correct3, color='blue', label='D-A OS*(10)')
    plt.plot(x, acc10, color='yellow', label='D-A acc10')
    plt.legend()  # 显示图例
    plt.show()
