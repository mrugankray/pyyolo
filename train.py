import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from library.utils import *
from collections import OrderedDict

data_transform = transforms.Compose([Rescale(240),RandomCrop(224),give_value(), Normalize(), ToTensor()])

'''
/content/pyyolo/dataset/images/mixed
/content/pyyolo/dataset/texts/mixed
/content/pyyolo/dataset/test/img
/content/pyyolo/dataset/test/txt
'''

##Loading Dataset##
transformed_train_dataset = yoloDataset(rootDirImg = '/content/pyyolo/dataset/images/mixed',
rootDirCoord = '/content/pyyolo/dataset/texts/mixed',transform=data_transform)

transformed_test_dataset = yoloDataset(rootDirImg = '/content/pyyolo/dataset/test/img',
rootDirCoord = '/content/pyyolo/dataset/test/txt',transform=data_transform)

#print(len(transformed_train_dataset))

len_dataset = len(transformed_train_dataset)
indices = list(range(len_dataset))
np.random.shuffle(indices)
split = int(np.floor(0.2*len_dataset))
train_idx = indices[split:]
val_idx = indices[:split]

##SAMPLER##
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

num_workers = 0
batch_size = 8

##Loading Data##
train_loader = DataLoader(transformed_train_dataset, num_workers = num_workers, batch_size = batch_size, sampler = train_sampler) 

val_loader = DataLoader(transformed_train_dataset, num_workers = num_workers, batch_size = batch_size, sampler = val_sampler)

test_loader = DataLoader(transformed_test_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cv1 = nn.Conv2d(1, 64, 3, stride = 1, padding = 1)
        self.cv2 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.cv3 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.cv4 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.cv5 = nn.Conv2d(128, 256, 3, stride = 1, padding = 1)
        self.cv6 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
        self.cv7 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
        self.cv8 = nn.Conv2d(256, 256, 3, stride = 1, padding = 1)
        self.cv9 = nn.Conv2d(256, 512, 3, stride = 1, padding = 1)
        self.cv10 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv11 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv12 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv13 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv14 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv15 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv16 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv17 = nn.Conv2d(512, 512, 3, stride = 1, padding = 1)
        self.cv18 = nn.Conv2d(512, 256, 3, stride = 1, padding = 1)
        self.cv19 = nn.Conv2d(256, 64, 3, stride = 1, padding = 1)
        self.cv20 = nn.Conv2d(64, 14, 3, stride = 1, padding = 1)

        self.dropout = nn.Dropout(0.25)

        self.maxpool = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = self.maxpool(F.relu(self.cv2(x)))
        x = self.dropout(F.relu(self.cv3(x)))
        x = self.maxpool(F.relu(self.cv4(x)))
        x = F.relu(self.cv5(x))
        x = F.relu(self.cv6(x))
        x = self.dropout(F.relu(self.cv7(x)))
        x = self.maxpool(F.relu(self.cv8(x)))
        x = F.relu(self.cv9(x))
        x = F.relu(self.cv10(x))
        x = self.dropout(F.relu(self.cv11(x)))
        x = self.maxpool(F.relu(self.cv12(x)))
        x = F.relu(self.cv13(x))
        x = F.relu(self.cv14(x))
        x = F.relu(self.cv15(x))
        x = F.relu(self.cv16(x))
        x = F.relu(self.cv17(x))
        x = F.relu(self.cv18(x))
        x = self.dropout(F.relu(self.cv19(x)))
        x = F.relu(self.cv20(x))

        return x

        '''
        (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (classifier): Sequential(
    (37): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): ReLU()
    (39): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1))
    (40): ReLU()
    (41): Conv2d(64, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        '''

'''model = models.vgg19(pretrained = False)
model.features[0] = nn.Conv2d(1, 64, 3, stride=(1,1), padding=(1,1))
model.features[36] = nn.Conv2d(512, 512, 3, stride=(1,1), padding=(1,1))

classifier = nn.Sequential(OrderedDict([
    ('37', nn.Conv2d(512, 256, 3, stride=1, padding=1)),
    ('38', nn.ReLU()),
    ('39', nn.Conv2d(256,64, 3, stride=1, padding=1)),
    ('40', nn.ReLU()),
    ('41', nn.Conv2d(64, 14, 3, stride=1, padding=1))
]))

model.classifier = classifier'''

'''model.fc = nn.Sequential(OrderedDict([
    ('fc1', nn.Conv2d())
]))'''

#print(model)
model = Net()
model = model.double()

#criterion#
criterion_coord = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')

criterion_img = nn.CrossEntropyLoss()

#Optimizer#
optimizer = optim.SGD(model.parameters(), lr=0.01)

if device == 'cuda':
    model = model.cuda()

#print(device)

trn_loss_list = []
val_loss_list = []
test_loss_list = []
val_min_loss = np.Inf

def train(epochs):
    for epoch in range(epochs):
        trn_loss = 0
        test_loss = 0
        val_loss = 0

        trn_running_loss = 0
        test_running_loss = 0
        val_running_loss = 0

        for trn_i, trn_sample in enumerate(train_loader):
            trn_img = trn_sample['image']
            output_tnsr = trn_sample['coord']
            grid_locate_x = trn_sample['grid_locate_x']
            grid_locate_y = trn_sample['grid_locate_y']
            print('output_tnsr_val shape',output_tnsr.shape)    
            print('image shape', trn_img.shape)      

            if device == 'cuda':
                trn_img = trn_img.type(torch.cuda.LongTensor)
                output_tnsr = output_tnsr.type(torch.cuda.LongTensor)
                trn_img = trn_img.cuda()
                output_tnsr = output_tnsr.cuda()
                pass

            #optizer to 0 grad
            optimizer.zero_grad()

            pred_tnsr = model(trn_img)
            print('pred tensor shape', pred_tnsr[0][0])

            #comuting losses
            '''if iou_anchor1 > iou_anchor2:
                loss_pc = criterion_coord(pred_tnsr[grid_locate_y][grid_locate_x][0], output_tnsr_val[grid_locate_y][grid_locate_x][0])

                loss_class = criterion_img(pred_tnsr[grid_locate_y][grid_locate_x][1], output_tnsr_val[grid_locate_y][grid_locate_x][1])

                loss_bounding_coord = criterion_coord([pred_tnsr[grid_locate_y][grid_locate_x][3], [grid_locate_y][grid_locate_x][4], [grid_locate_y][grid_locate_x][5] , [grid_locate_y][grid_locate_x][6]], [output_tnsr_val[grid_locate_y][grid_locate_x][3], output_tnsr_val[grid_locate_y][grid_locate_x][4], output_tns_testr[grid_locate_y][grid_locate_x][5], output_tnsr_val[grid_locate_y][grid_locate_x][6]])

            elif iou_anchor2 > iou_anchor1:
                loss_pc = criterion_coord(pred_tnsr[grid_locate_y][grid_locate_x][7], output_tnsr_val[grid_locate_y][grid_locate_x][7])

                loss_class = criterion_img(pred_tnsr[grid_locate_y][grid_locate_x][9], output_tnsr_val[grid_locate_y][grid_locate_x][9])

                loss_bounding_coord = criterion_coord([pred_tnsr[grid_locate_y][grid_locate_x][10], [grid_locate_y][grid_locate_x][11], [grid_locate_y][grid_locate_x][12] , [grid_locate_y][grid_locate_x][13]], [output_tnsr_val[grid_locate_y][grid_locate_x][10], output_tnsr_val[grid_locate_y][grid_locate_x][11], output_tns_testr[grid_locate_y][grid_locate_x][12], output_tnsr_val[grid_locate_y][grid_locate_x][13]])'''

            loss_pc = 0
            loss_class = 0
            loss_bounding_coord = 0
            for i in range(0,7):
                if output_tnsr[i][grid_locate_y][grid_locate_x][0] > -2:
                    loss_pc += criterion_coord(pred_tnsr[i][grid_locate_y][grid_locate_x][0], output_tnsr[i][grid_locate_y][grid_locate_x][0])

                    loss_class += criterion_img(pred_tnsr[i][grid_locate_y][grid_locate_x][1], output_tnsr[i][grid_locate_y][grid_locate_x][1])

                    loss_bounding_coord += criterion_coord([pred_tnsr[i][grid_locate_y][grid_locate_x][3],
                    pred_tnsr[i][grid_locate_y][grid_locate_x][4],pred_tnsr[i] [grid_locate_y][grid_locate_x][5] , pred_tnsr[grid_locate_y][grid_locate_x][6]], [output_tnsr[i][grid_locate_y][grid_locate_x][3],
                    output_tnsr[i][grid_locate_y][grid_locate_x][4],output_tnsr[i] [grid_locate_y][grid_locate_x][5] , output_tnsr[grid_locate_y][grid_locate_x][6]])

                else:
                    loss_pc += criterion_coord(output_tns_testr[i][grid_locate_y][grid_locate_x][7], pred_tnsr[i][grid_locate_y][grid_locate_x][7])

                    loss_class += criterion_img(pred_tnsr[i][grid_locate_y][grid_locate_x][9], output_tnsr[i][grid_locate_y][grid_locate_x][9])

                    loss_bounding_coord += criterion_coord([pred_tnsr[i][grid_locate_y][grid_locate_x][10],
                    pred_tnsr[i][grid_locate_y][grid_locate_x][11],pred_tnsr[i] [grid_locate_y][grid_locate_x][12] , pred_tnsr[grid_locate_y][grid_locate_x][13]], [output_tnsr[i][grid_locate_y][grid_locate_x][10],
                    output_tnsr[i][grid_locate_y][grid_locate_x][11],output_tnsr[i] [grid_locate_y][grid_locate_x][12] , output_tnsr[grid_locate_y][grid_locate_x][13]])


            total_loss = 0.33*loss_pc + 0.33*loss_class + 0.33*loss_bounding_coord

            # back propagation
            total_loss.backward()

            # optimizer
            optimize.step()

            trn_running_loss += total_loss.item() * batch_size

        else:
            with torch.no_grad():
                
                # sets model to evaluation mode
                model.eval()

                for val_img, val_sample in enumerate(val_loader):

                    val_img = val_sample['image']
                    output_tnsr_val = val_sample['coord']
                    grid_locate_x = val_sample['grid_locate_x']
                    grid_locate_y = val_sample['grid_locate_y']
                    #print('output_tnsr_val',output_tnsr_val)          

                    if device == 'cuda':
                        val_img = val_img.type(torch.cuda.LongTensor)
                        output_tnsr_val = output_tnsr_val.type(torch.cuda.LongTensor)
                        val_img = val_img.cuda()
                        output_tnsr_val = output_tnsr_val.cuda()
                        pass

                    pred_tnsr = model(val_img)

                    #comuting losses
                    loss_pc = 0
                    loss_class = 0
                    loss_bounding_coord = 0
                    for i in range(0,7):
                        if output_tnsr_val[i][grid_locate_y][grid_locate_x][0] > -2:
                            loss_pc += criterion_coord(pred_tnsr[i][grid_locate_y][grid_locate_x][0], output_tnsr_val[i][grid_locate_y][grid_locate_x][0])

                            loss_class += criterion_img(pred_tnsr[i][grid_locate_y][grid_locate_x][1], output_tnsr_val[i][grid_locate_y][grid_locate_x][1])

                            loss_bounding_coord += criterion_coord([pred_tnsr[i][grid_locate_y][grid_locate_x][3],
                            pred_tnsr[i][grid_locate_y][grid_locate_x][4],pred_tnsr[i] [grid_locate_y][grid_locate_x][5] , pred_tnsr[grid_locate_y][grid_locate_x][6]], [output_tnsr_val[i][grid_locate_y][grid_locate_x][3],
                            output_tnsr_val[i][grid_locate_y][grid_locate_x][4],output_tnsr_val[i] [grid_locate_y][grid_locate_x][5] , output_tnsr_val[grid_locate_y][grid_locate_x][6]])

                        else:
                            loss_pc += criterion_coord(output_tnsr_val[i][grid_locate_y][grid_locate_x][7], pred_tnsr[i][grid_locate_y][grid_locate_x][7])

                            loss_class += criterion_img(pred_tnsr[i][grid_locate_y][grid_locate_x][9], output_tnsr_val[i][grid_locate_y][grid_locate_x][9])

                            loss_bounding_coord += criterion_coord([pred_tnsr[i][grid_locate_y][grid_locate_x][10],
                            pred_tnsr[i][grid_locate_y][grid_locate_x][11],pred_tnsr[i] [grid_locate_y][grid_locate_x][12] , pred_tnsr[grid_locate_y][grid_locate_x][13]], [output_tnsr_val[i][grid_locate_y][grid_locate_x][10],
                            output_tnsr_val[i][grid_locate_y][grid_locate_x][11],output_tnsr_val[i] [grid_locate_y][grid_locate_x][12] , output_tnsr_val[grid_locate_y][grid_locate_x][13]])


                    total_loss_val = 0.33*loss_pc + 0.33*loss_class + 0.33*loss_bounding_coord

                    val_running_loss += total_loss_val.item() * batch_size

                for test_i, test_sample in enumerate(test_loader):
                    test_img = test_sample['image']
                    output_tnsr_test = test_sample['coord']
                    grid_locate_x = test_sample['grid_locate_x']
                    grid_locate_y = test_sample['grid_locate_y']
                    #print('output_tnsr_test',output_tnsr_test)          

                    if device == 'cuda':
                        test_img = test_img.type(torch.cuda.LongTensor)
                        output_tnsr_test = output_tnsr_test.type(torch.cuda.LongTensor)
                        test_img = test_img.cuda()
                        output_tnsr_test = output_tnsr_test.cuda()
                        pass

                    pred_tnsr = model(test_img)

                    #comuting losses
                    loss_pc = 0
                    loss_class = 0
                    loss_bounding_coord = 0
                    for i in range(0,7):
                        if output_tnsr_test[i][grid_locate_y][grid_locate_x][0] > -2:
                            loss_pc += criterion_coord(pred_tnsr[i][grid_locate_y][grid_locate_x][0], output_tnsr_test[i][grid_locate_y][grid_locate_x][0])

                            loss_class += criterion_img(pred_tnsr[i][grid_locate_y][grid_locate_x][1], output_tnsr_test[i][grid_locate_y][grid_locate_x][1])

                            loss_bounding_coord += criterion_coord([pred_tnsr[i][grid_locate_y][grid_locate_x][3],
                            pred_tnsr[i][grid_locate_y][grid_locate_x][4],pred_tnsr[i] [grid_locate_y][grid_locate_x][5] , pred_tnsr[grid_locate_y][grid_locate_x][6]], [output_tnsr_test[i][grid_locate_y][grid_locate_x][3],
                            output_tnsr_test[i][grid_locate_y][grid_locate_x][4],output_tnsr_test[i] [grid_locate_y][grid_locate_x][5] , output_tnsr_test[grid_locate_y][grid_locate_x][6]])

                        else:
                            loss_pc += criterion_coord(output_tnsr_test[i][grid_locate_y][grid_locate_x][7], pred_tnsr[i][grid_locate_y][grid_locate_x][7])

                            loss_class += criterion_img(pred_tnsr[i][grid_locate_y][grid_locate_x][9], output_tnsr_test[i][grid_locate_y][grid_locate_x][9])

                            loss_bounding_coord += criterion_coord([pred_tnsr[i][grid_locate_y][grid_locate_x][10],
                            pred_tnsr[i][grid_locate_y][grid_locate_x][11],pred_tnsr[i] [grid_locate_y][grid_locate_x][12] , pred_tnsr[grid_locate_y][grid_locate_x][13]], [output_tnsr_test[i][grid_locate_y][grid_locate_x][10],
                            output_tnsr_test[i][grid_locate_y][grid_locate_x][11],output_tnsr_test[i] [grid_locate_y][grid_locate_x][12] , output_tnsr_test[grid_locate_y][grid_locate_x][13]])


                    total_loss_test = 0.33*loss_pc + 0.33*loss_class + 0.33*loss_bounding_coord

                    test_running_loss += total_loss_test.item() * batch_size

            trn_loss = trn_running_loss/len(train_loader.dataset)
            val_loss = val_running_loss/len(val_loader.dataset)
            test_loss = test_running_loss/len(test_loader.dataset)
            trn_loss_list.append(trn_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)

            print(f'epochs: {epoch +1} / {epochs}, training loss: {trn_loss}, validation loss: {val_loss}, test_loss: {test_loss}')

            # set model to train mode
            model.train()

            if val_loss <= val_min_loss:
                print(f'validation loss has decreased {val_min_loss} ----> {val_loss}. Saving model')
                torch.save(model.state_dict(), 'model.pth')
                val_min_loss = val_loss

train(1)
'''x = 0
for trn_i, trn_sample in enumerate(train_loader):
    x = x+1
    trn_img = trn_sample['image']
    output_tnsr_val = trn_sample['coord']
    img_name = trn_sample['img_name']
    #print(img_name.shape())
    print(x)
    print('trn coord',img_name[0])
    break'''