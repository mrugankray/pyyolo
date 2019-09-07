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

##Loading Dataset##
transformed_train_dataset = yoloDataset(rootDirImg = '/content/dataset_colab/images/mixed',
rootDirCoord = '/content/dataset_colab/texts/mixed',transform=data_transform)

transformed_test_dataset = yoloDataset(rootDirImg = '/content/dataset_colab/test/img',
rootDirCoord = '/content/dataset_colab/test/txt',transform=data_transform)

#print(len(transformed_train_dataset))

len_dataset = len(transformed_train_dataset)
indices = list(range(len_dataset))
np.random.shuffle(indices)
split = int(np.floor(0.103*len_dataset))
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
train_on_gpu = torch.cuda.is_available()

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

#print(model)
model = Net()
model = model.float()

#criterion#
criterion_coord = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean')

criterion_img = nn.CrossEntropyLoss()

#Optimizer#
optimizer = optim.Adam(model.parameters(), lr=0.003)

if train_on_gpu:
    model = model.cuda()

#print(device)

trn_loss_list = []
val_loss_list = []
test_loss_list = []

def train(epochs):
    val_min_loss = np.Inf
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

            if train_on_gpu:
                trn_img = trn_img.type(torch.cuda.FloatTensor)
                trn_img = trn_img.cuda()
                pass

            #optizer to 0 grad
            optimizer.zero_grad()

            pred_tnsr = model(trn_img)

            loss_pc = 0
            loss_class = 0
            loss_bounding_coord = 0
            class_scores_global_list = []
            orginal_class_global_list = []
            pc_pred_global_list = []
            pc_inp_tnsr_global_list = []
            coord_pred_global_list = []
            coord_inp_global_list = []
            pred_tnsr = pred_tnsr.cpu().detach().numpy()
            grid_locate_x = grid_locate_x.detach().numpy()
            grid_locate_y = grid_locate_y.detach().numpy()
            output_tnsr = output_tnsr.detach().numpy()
            for i in range(0,len(trn_img)):

                if output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][0] > -2:

                    class_scores_list = np.array([pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][1], pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][2]])
                    class_scores_list = np.reshape(class_scores_list, (1,2))
                    class_scores_list = np.squeeze(class_scores_list)
                    orginal_class = np.array(output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][1])

                    # append to main lists
                    class_scores_global_list.append(class_scores_list)
                    
                    orginal_class_global_list.append(orginal_class)
                    
                    pc_pred_global_list.append(pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][0])
                    
                    pc_inp_tnsr_global_list.append(output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][0])
                    
                    coord_pred_global_list.append([pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][3],
                    pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][4],pred_tnsr[i] [grid_locate_y[i]][grid_locate_x[i]][5] , pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][6]])
                    
                    coord_inp_global_list.append([output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][3],
                    output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][4],output_tnsr[i] [grid_locate_y[i]][grid_locate_x[i]][5] , output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][6]])

                    continue

                else:
                    class_scores_list = np.array([pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][8], pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][9]])
                    class_scores_list = np.reshape(class_scores_list, (1,2))
                    class_scores_list = np.squeeze(class_scores_list)
                    orginal_class = np.array(output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][9])

                    # append to main lists
                    class_scores_global_list.append(class_scores_list)
                    
                    orginal_class_global_list.append(orginal_class)
                    
                    pc_pred_global_list.append(pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][7])
                    
                    pc_inp_tnsr_global_list.append(output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][7])
                    
                    coord_pred_global_list.append([pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][10],
                    pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][11],pred_tnsr[i] [grid_locate_y[i]][grid_locate_x[i]][12] , pred_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][13]])
                    
                    coord_inp_global_list.append([output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][10],
                    output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][11],output_tnsr[i] [grid_locate_y[i]][grid_locate_x[i]][12] , output_tnsr[i][grid_locate_y[i]][grid_locate_x[i]][13]])

                    continue

            # converting lists to numpy array
            for i in range(0,len(orginal_class_global_list)):
                orginal_class_global_list[i] = orginal_class_global_list[i]*50+100
                orginal_class_global_list[i] = int(orginal_class_global_list[i])
            pc_pred_global_list = np.array(pc_pred_global_list)
            pc_inp_tnsr_global_list = np.array(pc_inp_tnsr_global_list)
            class_scores_global_list = np.array(class_scores_global_list)
            orginal_class_global_list = np.array(orginal_class_global_list)
            coord_pred_global_list = np.array(coord_pred_global_list)
            coord_inp_global_list = np.array(coord_inp_global_list)

            #reshaping numpy array
            pc_pred_global_list = np.reshape(pc_pred_global_list,(len(trn_img), 1)) #size becomes 8,1
            pc_inp_tnsr_global_list = np.reshape(pc_inp_tnsr_global_list,(len(trn_img), 1))

            #converting numpy array to tensors
            pc_pred_global_list = torch.from_numpy(pc_pred_global_list)
            pc_pred_global_list.requires_grad_(True)
            pc_inp_tnsr_global_list = torch.from_numpy(pc_inp_tnsr_global_list)
            class_scores_global_list = torch.from_numpy(class_scores_global_list)
            class_scores_global_list.requires_grad_(True)
            orginal_class_global_list = torch.from_numpy(orginal_class_global_list)
            coord_pred_global_list = torch.from_numpy(coord_pred_global_list)
            coord_pred_global_list.requires_grad_(True)
            coord_inp_global_list = torch.from_numpy(coord_inp_global_list)

            if train_on_gpu:
                pc_pred_global_list = pc_pred_global_list.type(torch.cuda.FloatTensor)
                pc_inp_tnsr_global_list = pc_inp_tnsr_global_list.type(torch.cuda.FloatTensor)
                coord_pred_global_list = coord_pred_global_list.type(torch.cuda.FloatTensor)
                coord_inp_global_list = coord_inp_global_list.type(torch.cuda.FloatTensor)
                pc_pred_global_list = pc_pred_global_list.cuda()
                pc_inp_tnsr_global_list = pc_inp_tnsr_global_list.cuda()
                class_scores_global_list = class_scores_global_list.cuda()
                orginal_class_global_list = orginal_class_global_list.cuda()
                coord_pred_global_list = coord_pred_global_list.cuda()
                coord_inp_global_list = coord_inp_global_list.cuda()
                pass

            loss_pc = criterion_coord(pc_pred_global_list, pc_inp_tnsr_global_list)
            loss_class = criterion_img(class_scores_global_list, orginal_class_global_list)
            loss_bounding_coord = criterion_coord(coord_pred_global_list, coord_inp_global_list)

            total_loss = 0.50*loss_pc + 0.50*loss_class + 0.50*loss_bounding_coord

            # back propagation
            total_loss.backward()

            # optimizer
            optimizer.step()

            trn_running_loss += total_loss.item() * len(trn_img)

        else:
            with torch.no_grad():
                
                # sets model to evaluation mode
                model.eval()

                for val_img, val_sample in enumerate(val_loader):
                    #print(len(val_sample))
                    val_img = val_sample['image']
                    output_tnsr_val = val_sample['coord']
                    grid_locate_x_val = val_sample['grid_locate_x']
                    grid_locate_y_val = val_sample['grid_locate_y']
                    #print('output_tnsr_val',output_tnsr_val)          

                    if train_on_gpu:
                        val_img = val_img.type(torch.cuda.FloatTensor)
                        #output_tnsr_val = output_tnsr_val.type(torch.cuda.FloatTensor)
                        val_img = val_img.cuda()
                        #output_tnsr_val = output_tnsr_val.cuda()
                        pass

                    pred_tnsr = model(val_img)

                    #comuting losses
                    loss_pc = 0
                    loss_class = 0
                    loss_bounding_coord = 0
                    class_scores_global_list_val = []
                    orginal_class_global_list_val = []
                    pc_pred_global_list_val = []
                    pc_inp_tnsr_global_list_val = []
                    coord_pred_global_list_val = []
                    coord_inp_global_list_val = []
                    pred_tnsr = pred_tnsr.cpu().detach().numpy()
                    grid_locate_x_val = grid_locate_x_val.detach().numpy()
                    grid_locate_y_val = grid_locate_y_val.detach().numpy()
                    output_tnsr_val = output_tnsr_val.detach().numpy()
                    for i in range(0,len(val_img)):
                        if output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][0] > -2:

                            class_scores_list_val = np.array([pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][1], pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][2]])
                            class_scores_list_val = np.reshape(class_scores_list_val, (1,2))
                            class_scores_list_val = np.squeeze(class_scores_list_val)
                            orginal_class_val = np.array(output_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][1])

                            # append to main lists
                            class_scores_global_list_val.append(class_scores_list_val)
                            
                            orginal_class_global_list_val.append(orginal_class_val)
                            
                            pc_pred_global_list_val.append(pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][0])
                            
                            pc_inp_tnsr_global_list_val.append(output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][0])
                            
                            coord_pred_global_list_val.append([pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][3],
                            pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][4],pred_tnsr[i] [grid_locate_y_val[i]][grid_locate_x_val[i]][5] , pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][6]])
                            
                            coord_inp_global_list_val.append([output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][3],
                            output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][4],output_tnsr_val[i] [grid_locate_y_val[i]][grid_locate_x_val[i]][5] , output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][6]])

                            continue

                        else:

                            class_scores_list_val = np.array([pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][8], pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][9]])
                            class_scores_list_val = np.reshape(class_scores_list_val, (1,2))
                            class_scores_list_val = np.squeeze(class_scores_list_val)
                            orginal_class_val = np.array(output_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][9])

                            # append to main lists
                            class_scores_global_list_val.append(class_scores_list_val)
                            
                            orginal_class_global_list_val.append(orginal_class_val)
                            
                            pc_pred_global_list_val.append(pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][7])
                            
                            pc_inp_tnsr_global_list_val.append(output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][7])
                            
                            coord_pred_global_list_val.append([pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][10],
                            pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][11],pred_tnsr[i] [grid_locate_y_val[i]][grid_locate_x_val[i]][12] , pred_tnsr[i][grid_locate_y_val[i]][grid_locate_x_val[i]][13]])
                            
                            coord_inp_global_list_val.append([output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][10],
                            output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][11],output_tnsr_val[i] [grid_locate_y_val[i]][grid_locate_x_val[i]][12] , output_tnsr_val[i][grid_locate_y_val[i]][grid_locate_x_val[i]][13]])

                            continue

                    # converting lists to numpy array
                    for i in range(0,len(orginal_class_global_list_val)):
                        orginal_class_global_list_val[i] = orginal_class_global_list_val[i]*50+100
                        orginal_class_global_list_val[i] = int(orginal_class_global_list_val[i])
                    pc_pred_global_list_val = np.array(pc_pred_global_list_val)
                    pc_inp_tnsr_global_list_val = np.array(pc_inp_tnsr_global_list_val)
                    class_scores_global_list_val = np.array(class_scores_global_list_val)
                    orginal_class_global_list_val = np.array(orginal_class_global_list_val)
                    coord_pred_global_list_val = np.array(coord_pred_global_list_val)
                    coord_inp_global_list_val = np.array(coord_inp_global_list_val)

                    #reshaping numpy array
                    pc_pred_global_list_val = np.reshape(pc_pred_global_list_val,(len(val_img), 1))
                    pc_inp_tnsr_global_list_val = np.reshape(pc_inp_tnsr_global_list_val,(len(val_img), 1))

                    #converting numpy array to tensors
                    pc_pred_global_list_val = torch.from_numpy(pc_pred_global_list_val)
                    pc_inp_tnsr_global_list_val = torch.from_numpy(pc_inp_tnsr_global_list_val)
                    class_scores_global_list_val = torch.from_numpy(class_scores_global_list_val)
                    orginal_class_global_list_val = torch.from_numpy(orginal_class_global_list_val)
                    coord_pred_global_list_val = torch.from_numpy(coord_pred_global_list_val)
                    coord_inp_global_list_val = torch.from_numpy(coord_inp_global_list_val)

                    if train_on_gpu:
                        pc_pred_global_list_val = pc_pred_global_list_val.type(torch.cuda.FloatTensor)
                        pc_inp_tnsr_global_list_val = pc_inp_tnsr_global_list_val.type(torch.cuda.FloatTensor)
                        coord_pred_global_list_val = coord_pred_global_list_val.type(torch.cuda.FloatTensor)
                        coord_inp_global_list_val = coord_inp_global_list_val.type(torch.cuda.FloatTensor)
                        pc_pred_global_list_val = pc_pred_global_list_val.cuda()
                        pc_inp_tnsr_global_list_val = pc_inp_tnsr_global_list_val.cuda()
                        class_scores_global_list_val = class_scores_global_list_val.cuda()
                        orginal_class_global_list_val = orginal_class_global_list_val.cuda()
                        coord_pred_global_list_val = coord_pred_global_list_val.cuda()
                        coord_inp_global_list_val = coord_inp_global_list_val.cuda()
                        pass


                    loss_pc = criterion_coord(pc_pred_global_list_val, pc_inp_tnsr_global_list_val)
                    loss_class = criterion_img(class_scores_global_list_val, orginal_class_global_list_val)
                    loss_bounding_coord = criterion_coord(coord_pred_global_list_val, coord_inp_global_list_val)

                    total_loss_val = 0.50*loss_pc + 0.50*loss_class + 0.50*loss_bounding_coord

                    val_running_loss += total_loss_val.item() * len(val_img)

                for test_i, test_sample in enumerate(test_loader):
                    test_img = test_sample['image']
                    output_tnsr_test = test_sample['coord']
                    grid_locate_x_test = test_sample['grid_locate_x']
                    grid_locate_y_test = test_sample['grid_locate_y']
                    #print('output_tnsr_test',output_tnsr_test)          

                    if train_on_gpu:
                        test_img = test_img.type(torch.cuda.FloatTensor)
                        #output_tnsr_test = output_tnsr_test.type(torch.cuda.FloatTensor)
                        test_img = test_img.cuda()
                        #output_tnsr_test = output_tnsr_test.cuda()
                        pass

                    pred_tnsr = model(test_img)

                    #comuting losses
                    loss_pc = 0
                    loss_class = 0
                    loss_bounding_coord = 0
                    class_scores_global_list_test = []
                    orginal_class_global_list_test = []
                    pc_pred_global_list_test = []
                    pc_inp_tnsr_global_list_test = []
                    coord_pred_global_list_test = []
                    coord_inp_global_list_test = []
                    pred_tnsr = pred_tnsr.cpu().detach().numpy()
                    grid_locate_x_test = grid_locate_x_test.detach().numpy()
                    grid_locate_y_test = grid_locate_y_test.detach().numpy()
                    output_tnsr_test = output_tnsr_test.detach().numpy()
                    for i in range(0,len(test_img)):
                        if output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][0] > -2:

                            class_scores_list_test = np.array([pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][1], pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][2]])
                            class_scores_list_test = np.reshape(class_scores_list_test, (1,2))
                            class_scores_list_test = np.squeeze(class_scores_list_test)
                            orginal_class_test = np.array(output_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][1])

                            # append to main lists
                            class_scores_global_list_test.append(class_scores_list_test)
                            
                            orginal_class_global_list_test.append(orginal_class_test)
                            
                            pc_pred_global_list_test.append(pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][0])
                            
                            pc_inp_tnsr_global_list_test.append(output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][0])
                            
                            coord_pred_global_list_test.append([pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][3],
                            pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][4],pred_tnsr[i] [grid_locate_y_test[i]][grid_locate_x_test[i]][5] , pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][6]])
                            
                            coord_inp_global_list_test.append([output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][3],
                            output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][4],output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][5] , output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][6]])

                            continue

                        else:

                            class_scores_list_test = np.array([pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][8], pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][9]])
                            class_scores_list_test = np.reshape(class_scores_list_test, (1,2))
                            class_scores_list_test = np.squeeze(class_scores_list_test)
                            orginal_class_test = np.array(output_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][9])

                            # append to main lists
                            class_scores_global_list_test.append(class_scores_list_test)
                            
                            orginal_class_global_list_test.append(orginal_class_test)
                            
                            pc_pred_global_list_test.append(pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][7])
                            
                            pc_inp_tnsr_global_list_test.append(output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][7])
                            
                            coord_pred_global_list_test.append([pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][10],
                            pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][11],pred_tnsr[i] [grid_locate_y_test[i]][grid_locate_x_test[i]][12] , pred_tnsr[i][grid_locate_y_test[i]][grid_locate_x_test[i]][13]])
                            
                            coord_inp_global_list_test.append([output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][10],
                            output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][11],output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][12] , output_tnsr_test[i][grid_locate_y_test[i]][grid_locate_x_test[i]][13]])

                            continue
                    
                    #orginal_class_global_list_test = list(map(int, orginal_class_global_list_test*50 + 100))
                    for i in range(0,len(orginal_class_global_list_test)):
                        orginal_class_global_list_test[i] = orginal_class_global_list_test[i]*50+100
                        orginal_class_global_list_test[i] = int(orginal_class_global_list_test[i])
                    # converting lists to numpy array
                    pc_pred_global_list_test = np.array(pc_pred_global_list_test)
                    pc_inp_tnsr_global_list_test = np.array(pc_inp_tnsr_global_list_test)
                    class_scores_global_list_test = np.array(class_scores_global_list_test)
                    orginal_class_global_list_test = np.array(orginal_class_global_list_test)
                    coord_pred_global_list_test = np.array(coord_pred_global_list_test)
                    coord_inp_global_list_test = np.array(coord_inp_global_list_test)

                    #reshaping numpy array
                    pc_pred_global_list_test = np.reshape(pc_pred_global_list_test,(len(test_img), 1))
                    pc_inp_tnsr_global_list_test = np.reshape(pc_inp_tnsr_global_list_test,(len(test_img), 1))

                    #converting numpy array to tensors
                    pc_pred_global_list_test = torch.from_numpy(pc_pred_global_list_test)
                    pc_inp_tnsr_global_list_test = torch.from_numpy(pc_inp_tnsr_global_list_test)
                    class_scores_global_list_test = torch.from_numpy(class_scores_global_list_test)
                    orginal_class_global_list_test = torch.from_numpy(orginal_class_global_list_test)
                    coord_pred_global_list_test = torch.from_numpy(coord_pred_global_list_test)
                    coord_inp_global_list_test = torch.from_numpy(coord_inp_global_list_test)

                    if train_on_gpu:
                        pc_pred_global_list_test = pc_pred_global_list_test.type(torch.cuda.FloatTensor)
                        pc_inp_tnsr_global_list_test = pc_inp_tnsr_global_list_test.type(torch.cuda.FloatTensor)
                        #class_scores_global_list_test = class_scores_global_list_test.type(torch.cuda.FloatTensor)
                        #orginal_class_global_list_val = orginal_class_global_list_test.type(torch.cuda.FloatTensor)
                        coord_pred_global_list_test = coord_pred_global_list_test.type(torch.cuda.FloatTensor)
                        coord_inp_global_list_test = coord_inp_global_list_test.type(torch.cuda.FloatTensor)
                        pc_pred_global_list_test = pc_pred_global_list_test.cuda()
                        pc_inp_tnsr_global_list_test = pc_inp_tnsr_global_list_test.cuda()
                        class_scores_global_list_test = class_scores_global_list_test.cuda()
                        orginal_class_global_list_test = orginal_class_global_list_test.cuda()
                        coord_pred_global_list_test = coord_pred_global_list_test.cuda()
                        coord_inp_global_list_test = coord_inp_global_list_test.cuda()
                        pass

                    loss_pc = criterion_coord(pc_pred_global_list_test, pc_inp_tnsr_global_list_test)
                    loss_class = criterion_img(class_scores_global_list_test, orginal_class_global_list_test)
                    loss_bounding_coord = criterion_coord(coord_pred_global_list_test, coord_inp_global_list_test)

                    total_loss_test = 0.50*loss_pc + 0.50*loss_class + 0.50*loss_bounding_coord

                    test_running_loss += total_loss_test.item() * len(test_img)

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

train(50)