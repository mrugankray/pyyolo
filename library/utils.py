import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import re
from torchvision import transforms
import matplotlib.image as mpimg

def load_img(img = None,txt_file = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-1.txt', img_file = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-1.jpg'):
    #txt_dog_name_arr = os.listdir(txt_dog_fold)
    #print(name_txt_dog)
    #img_dog_name_arr = []
    #print(txt_dog_name_arr[i])
    if img == None:
        frame = cv2.imread(img_file)
    else:
        frame = img
    #frame = cv2.resize(frame, (300,300), interpolation = cv2.INTER_AREA)
    '''plt.imshow(frame,cmap='gray')
    plt.show()'''
    return frame
    #print(frame.shape)
    '''cv2.imshow('dog', frame)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.destroyAllWindows()'''

#load_img()

def readCoord(path_inp):
    with open(path_inp, 'r') as f:
        #print(path_inp)
        lines = f.readlines()
        lines = re.split(r'\s', lines[1])
        lines = list(map(int, lines))
        return np.array(lines)
        #print(lines)
        #[40, 234, 35, 213]

#readCoord()

def findCentroid(xmin,xmax,ymin,ymax):
    '''if xmin == None:
        coord = readCoord(path = path)
        #xmin, ymin = coord[0]/2, coord[2]/2
        xcenter , ycenter = (coord[1]+coord[0])/2 , (coord[3]+coord[2])/2
        #print(type(xmax))
        centroid_arr = np.array([xcenter, ycenter])
        #centroid_arr.reshape(1,-1)
        #centroid_arr = np.squeeze(centroid_arr)
        #print(centroid_arr)
        #[ 20.  117.   17.5 106.5]'''
    #else:
    xcenter , ycenter = (xmin+xmax)/2 , (ymin+ymax)/2
    centroid_arr = np.array([xcenter, ycenter])
    return centroid_arr
    #print(coord)
    #[40, 234, 35, 213]

#findCentroid(50,60,70,80)

def showCenter(img = None ,xmin=None, xmax=None, ymin=None, ymax=None ,Imgpath = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-27.jpg',TxtrootDirCoordPath = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-27.txt'):
    #gridFrame = grid(path = path)
    if xmin == None:
        frame = cv2.imread(Imgpath,0)
        print('iimage shape xmin',frame.shape)
        centroidArr = findCentroid(path = TxtrootDirCoordPath)
        frame = cv2.circle(frame, (int(centroidArr[0]), int(centroidArr[1])), 2, (0,0,255), -1)
        coord = readCoord(path = TxtrootDirCoordPath)
        frame = cv2.rectangle(frame, (coord[0], coord[2]), (coord[1], coord[3]), (0,0,255), 2)
    else:
        frame = img
        print('iimage shape img',frame.shape)
        centroidArr = findCentroid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        frame = cv2.circle(frame, (int(centroidArr[0]), int(centroidArr[1])), 2, (0,0,255), -1)
        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)

    #print(centroidArr)
    #[137. 124.]
    #print(coord)
    #[40, 234, 35, 213]
    plt.imshow(frame, cmap='gray')
    plt.show()


def grid(path = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-1.jpg',height = 16, width = 16):
    tx = 0
    ty = 0
    bx = 16
    by = 16
    frame = load_img(img_file = path)
    frame_copy = np.copy(frame)
    frame_copy = cv2.resize(frame_copy, (224,224), interpolation = cv2.INTER_AREA)
    for j in range(0,224,height):
        for i in range(0,224,width):
            tx = tx + i
            ty = ty + j
            bx = bx + i
            by = by + j
            frame_copy = cv2.rectangle(frame_copy,(tx,ty),(bx,by), (0,255,0), 1)
            tx = 0
            ty = 0
            bx = 16
            by = 16
        if j == 224:
            break
    plt.imshow(frame_copy, cmap='gray')
    plt.show()
    return frame_copy
#grid()

def find_grid(xcenter, ycenter):
    result = {'xcenter':0, 'ycenter':0}
    countx = 0
    county = 0
    gridx = 0
    gridy = 0
    for i in range(int(16), int(240), int(16)):
        #print(i)
        gridx = gridx + 1
        gridy = gridy + 1
        resx = xcenter - i
        resy = ycenter - i
        if resx < 0 and countx == 0:
            result['xcenter'] = gridx
            countx = countx + 1
            #print(resx)
        if resy < 0 and county == 0:
            result['ycenter'] = gridy
            county = county + 1
            #print(resy)
    #print(result)
    return result

find_grid(10, 10)
#{'xcenter': 1, 'ycenter': 1}

def find_grid_coord(xcenter, ycenter, x_min, x_max , y_min , y_max ):
    grid = find_grid(xcenter = xcenter, ycenter = ycenter)
    gridx = grid['xcenter']
    gridy = grid['ycenter']
    #print(gridx)
    #print(gridy)
    gridx_pix_coord = (gridx - 1) * 16 #coz gridx starts from 1 i want it to start from 0
    gridy_pix_coord = (gridy - 1) * 16
    #print(gridx_pix_coord, gridy_pix_coord)
    xcoord_wrt_grid = xcenter - gridx_pix_coord
    ycoord_wrt_grid = ycenter - gridy_pix_coord
    #if x_min != None:
    w, h = x_max - x_min, y_max - y_min
    w_wrt_grid, h_wrt_grid = w/16, h/16
    #print(xcoord_wrt_grid, ycoord_wrt_grid, w_wrt_grid, h_wrt_grid)
    anchor1_w,anchor1_h = (w_wrt_grid*0.8), (h_wrt_grid*1.2)
    anchor2_w,anchor2_h = (w_wrt_grid*1.2), (h_wrt_grid*0.8)
    convert_list = np.array([xcoord_wrt_grid, ycoord_wrt_grid, w_wrt_grid, h_wrt_grid, anchor1_w, anchor1_h, anchor2_w, anchor2_h])
    return(convert_list)

def find_area(xmin, xmax, ymin, ymax):
    w = xmax - xmin
    h = ymax - ymin
    area = w*h
    return area

def find_iou(bounding_box1, bounding_box2):
    xcenter_box1, ycenter_box1, width_box1, height_box1 = bounding_box1[0], bounding_box1[1], bounding_box1[2], bounding_box1[3]

    #print('xcenter_box1, ycenter_box1, width_box1, height_box1',xcenter_box1, ycenter_box1, width_box1, height_box1)

    xmin_box1, xmax_box1, ymin_box1, ymax_box1 = xcenter_box1 - (width_box1/2), xcenter_box1 + (width_box1/2), ycenter_box1 - (height_box1/2), ycenter_box1 + (height_box1/2) 

    #print('xmin_box1, xmax_box1, ymin_box1, ymax_box1', xmin_box1, xmax_box1, ymin_box1, ymax_box1)

    xcenter_box2, ycenter_box2, width_box2, height_box2 = bounding_box2[0], bounding_box2[1], bounding_box2[2], bounding_box2[3]

    #print('xcenter_box2, ycenter_box2, width_box2, height_box2', xcenter_box2, ycenter_box2, width_box2, height_box2)

    xmin_box2, xmax_box2, ymin_box2, ymax_box2 = xcenter_box2 - (width_box2/2), xcenter_box2 + (width_box2/2), ycenter_box2 - (height_box2/2), ycenter_box2 + (height_box2/2)

    #print('xmin_box2, xmax_box2, ymin_box2, ymax_box2', xmin_box2, xmax_box2, ymin_box2, ymax_box2)

    intersection_rect_coord_xmin, intersection_rect_coord_xmax, intersection_rect_coord_ymin, intersection_rect_coord_ymax = max(xmin_box1, xmin_box2), min(xmax_box1, xmax_box2), max(ymin_box1, ymin_box2), min(ymax_box1, ymax_box2)

    interArea = max(0, intersection_rect_coord_xmax - intersection_rect_coord_xmin + 1) * max(0, intersection_rect_coord_ymax - intersection_rect_coord_ymin + 1)

    intersection_rect_coord = np.array([intersection_rect_coord_xmin, intersection_rect_coord_xmax, intersection_rect_coord_ymin, intersection_rect_coord_ymax])

    bounding_box1_area = ((xmax_box1 - xmin_box1) + 1) * ((ymax_box1 - ymin_box1) + 1)
    bounding_box2_area = ((xmax_box2 - xmin_box2) + 1) * ((ymax_box2 - ymin_box2) + 1)

    '''print('intersection_rect_coord_xmin',intersection_rect_coord_xmin)
    print('intersection_rect_coord_xmax',intersection_rect_coord_xmax)
    print('intersection_rect_coord_ymin', intersection_rect_coord_ymin)
    print('intersection_rect_coord_ymax',intersection_rect_coord_ymax)'''

    iou = interArea/(bounding_box1_area + bounding_box2_area - interArea)
    return iou

#find_iou([21,30,50, 100], [20,30,50,100])

#find_grid_coord(17, 16, x_min=50, x_max=150, y_min=100, y_max=300)

def output_tensor(xcenter, ycenter, input_array):
    grid = find_grid(xcenter = xcenter, ycenter = ycenter)
    gridx = grid['xcenter'] - 1
    gridy = grid['ycenter'] - 1
    #print(gridx)
    numpy_tensor = np.zeros((14, 14, 14))
    numpy_tensor[gridy][gridx] = input_array
    #print(numpy_tensor[gridy][gridx])
    return numpy_tensor
    #print(gridx, gridy)
    #print(numpy_tensor[gridy][0:])


#output_tensor(xcenter = 200, ycenter = 150, input_array = [1,1,1,1,1,1,1,1,1,1,1,1,1,1])

class give_value(object):
    def __call__(self, sample):#img_name, xmin, xmax, ymin, ymax

        img_name, xmin, xmax, ymin, ymax = sample['img_name'], sample['coord'][0], sample['coord'][1], sample['coord'][2], sample['coord'][3]

        img_name = img_name
        #print('img name',img_name)
        #print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)

        if img_name.find('person') != -1:
            img_class = 1
        else:
            img_class = 0

        centroid = findCentroid(xmin = xmin ,xmax = xmax, ymin = ymin, ymax = ymax)

        #print('centrod', centroid)

        xcenter, ycenter = centroid[0], centroid[1]

        grid_locate = find_grid(xcenter=xcenter, ycenter=ycenter)
        grid_locate_x, grid_locate_y = grid_locate['xcenter'], grid_locate['ycenter'] 

        #find_grid_coord will give an array having center wrt grid and w,h and anchor boxes
        grid_coord_wrt_grid = find_grid_coord(xcenter=xcenter, ycenter=ycenter, x_min = xmin ,x_max = xmax, y_min = ymin, y_max = ymax)
        #print('grid_coord_wrt_grid',grid_coord_wrt_grid)

        anchor1_w, anchor1_h  = grid_coord_wrt_grid[4], grid_coord_wrt_grid[5]
        anchor2_w, anchor2_h  = grid_coord_wrt_grid[6], grid_coord_wrt_grid[7]

        #print('anchor1_w, anchor1_h, anchor2_w, anchor2_h', anchor1_w, anchor1_h, anchor2_w, anchor2_h)

        iou_anchor1 = find_iou([grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor1_w, anchor1_h], [grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], grid_coord_wrt_grid[2], grid_coord_wrt_grid[3]])

        iou_anchor2 = find_iou([grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor2_w, anchor2_h], [grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], grid_coord_wrt_grid[2], grid_coord_wrt_grid[3]])

        #print('iou_anchor1, iou_anchor2', iou_anchor1, iou_anchor2)

        if iou_anchor1 > iou_anchor2:
            if img_class == 1:
                input_vector = [1, 1, 0, grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor1_w, anchor1_h, 0, 0, 0, 0, 0, 0, 0]
            else:
                input_vector = [1, 0, 1, grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor1_w, anchor1_h, 0, 0, 0, 0, 0, 0, 0]

        if iou_anchor2 > iou_anchor1:
            if img_class == 1:
                input_vector = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor2_w, anchor2_h]
            else:
                input_vector = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor2_w, anchor2_h]
        if iou_anchor1 == iou_anchor2:
            if img_class == 1:
                input_vector = [1, 1, 0, grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor1_w, anchor1_h, 0, 0, 0, 0, 0, 0, 0]
            else:
                input_vector = [1, 0, 1, grid_coord_wrt_grid[0], grid_coord_wrt_grid[1], anchor1_w, anchor1_h, 0, 0, 0, 0, 0, 0, 0]
        #print(iou_anchor1, iou_anchor2)
        #print('input_vector', input_vector)
        #print('grid_coord_wrt_grid[0]', grid_coord_wrt_grid[0])
        output_tnsr = output_tensor(xcenter = xcenter, ycenter = ycenter, input_array = input_vector)
        output_tnsr = np.array(output_tnsr)
        #print('output tensor shape', output_tnsr.shape)
        #print(output_tnsr)
        #print(output_tnsr[grid_locate_y][grid_locate_x])
        return {'image': sample['image'], 'coord': output_tnsr, 'img_name': sample['img_name'], 'grid_locate_x':grid_locate_x - 1,'grid_locate_y':grid_locate_y - 1, 'img_class': img_class}

#give_value_obj = give_value(img_name = None, xmin = None, xmax = None, ymin = None, ymax = None)

class yoloDataset(Dataset):
    def __init__(self, rootDirImg, rootDirCoord, transform = None):
        self.rootDirImg = rootDirImg
        self.rootDirCoord = rootDirCoord
        self.transform = transform
        self.give_value = give_value

    def __len__(self):
        return(len(os.listdir(self.rootDirImg)))

    def __getitem__(self, idx):
        tempName = os.listdir(self.rootDirImg)[idx]
        imgName = os.path.join(self.rootDirImg, tempName)
        '''print(tempName)
        print(imgName)
        person10.jpg
        /media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/github/dataset_colab/images/mixed/person10.jpg'''

        frame = mpimg.imread(imgName,0)
        #print(frame)
        if frame.shape[2] == 4:
            frame = frame[:,:,0:3]
        if tempName.find('.jpg') >= 0:
            #print('1st',tempName.find('jpg'))
            coord = readCoord(path_inp = os.path.join(self.rootDirCoord, tempName.replace('.jpg','.txt')))
        elif tempName.find('.jpeg') >= 0:
            #print('2nd',tempName.find('jpeg'))
            coord = readCoord(path_inp = os.path.join(self.rootDirCoord, tempName.replace('.jpeg','.txt')))
        elif tempName.find('.png') >= 0:
            coord = readCoord(path_inp = os.path.join(self.rootDirCoord, tempName.replace('.png','.txt')))
        #print(tempName)
        #print(coord)

        sample = {'image': frame, 'coord': coord, 'img_name': tempName,'grid_locate_x':0,'grid_locate_y':0,'img_class': None}

        '''self.img = sample['image']
        self.coord = sample['coord']
        self.img_name = sample['img_name']'''

        #print('sample',sample['img_name'])

        '''try:

            tnsr = give_value(img_name = sample['img_name'], xmin = sample['coord'][0], xmax = sample['coord'][1], ymin = sample['coord'][2], ymax = sample['coord'][3])
        except:
            print('wait')'''

        #print('output tensor', tnsr.shape)

        if self.transform:
            sample = self.transform(sample)
            #print('before transform',sample['coord'])
            #print('output tensor', tnsr.shape)
            #sample = {'image': frame, 'coord': tnsr, 'img_name': tempName}

        #sample = {'image': frame, 'coord': coord}
        '''print('yolodataset',sample['coord'])
        print('name of img', imgName)
        print('directory', os.listdir(self.rootDirImg))''' 
        return sample

#print(obj_det)
#49

#see random pic#

'''for i in range(0,num_of_pics):
    idx = np.random.randint(0, len(obj_det))
    sample = obj_det[idx]
    #print(sample)
    showCenter(img = sample['image'], xmin = sample['coord'][0],xmax = sample['coord'][1],ymin = sample['coord'][2],ymax = sample['coord'][3])
    frame = load_img(img = sample['image'])
    plt.imshow(frame, cmap = 'gray')
    plt.show()'''

class Normalize(object):
    def __call__(self, sample):
        image, coord = sample['image'], sample['coord']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image/255.0
        for i in range(0,14):           
            coord[i] = (coord[i] - 100)/50.0
        #print('Normalize', sample['coord'])
        return {'image': image, 'coord': coord, 'img_name': sample['img_name'], 'grid_locate_x':sample['grid_locate_x'],'grid_locate_y':sample['grid_locate_y'],
        'img_class': sample['img_class']}

class Rescale(object):
    def __init__(self, outputSize):
        assert isinstance(outputSize, (int, tuple))
        self.outputSize = outputSize
    
    def __call__(self, sample):
        image, coord = sample['image'], sample['coord']
        h, w = image.shape[:2]
        
        if isinstance(self.outputSize, int):
            if h > w:
                ratio = h/w
                newH, newW = self.outputSize * ratio, self.outputSize
            else:
                ratio = w/h
                newH, newW = self.outputSize, self.outputSize * ratio
        
        else:
            newH, newW = self.outputSize
        
        image = cv2.resize(image, (int(newW), int(newH)), interpolation = cv2.INTER_AREA)

        wRatio = newW/w
        hRatio = newH/h 

        #print('coord[0]', coord[0]) 

        coord[0], coord[2] = coord[0]*(wRatio), coord[2]*(hRatio)
        coord[1], coord[3] = coord[1]*(wRatio), coord[3]*(hRatio)

        '''print('rescale',coord)
        print('wRAtio', wRatio)
        print('hRatio', hRatio)
        print('newW', newW)
        print('w', w)'''

        return {'image': image, 'coord': coord, 'img_name': sample['img_name'],'grid_locate_x':sample['grid_locate_x'],'grid_locate_y':sample['grid_locate_y'],'img_class': sample['img_class']}

class RandomCrop(object):
    def __init__(self, outputSize):
        assert isinstance(outputSize, (int, tuple))
        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
            self.outputSize = outputSize

    def __call__(self, sample):
        image, coord = sample['image'], sample['coord']

        h, w = image.shape[:2]

        newW, newH = self.outputSize

        x = np.random.randint(0, w - newW)
        y = np.random.randint(0, h - newH)

        image = image[y:y+newH, x:x+newW]

        coord[0], coord[1], coord[2], coord[3] = coord[0] - x, coord[1] - x, coord[2] - y, coord[3] - y

        #print('randcrop', type(coord[0]))
        #print('randomcrop',coord)
        return {'image': image, 'coord': coord, 'img_name': sample['img_name'],'grid_locate_x':sample['grid_locate_x'],'grid_locate_y':sample['grid_locate_y'], 'img_class': sample['img_class']}

class ToTensor(object):
    def __call__(self, sample):
        image, coord = sample['image'], sample['coord']
        #print('1st image shape',image.shape)
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            #print('2nd image shape',image.shape)
        
        image = image.transpose((2,0,1))
        #print('image shape after transpose', image.shape)

        return {'image': image, 'coord': coord, 'img_name': sample['img_name'],'grid_locate_x':sample['grid_locate_x'],'grid_locate_y':sample['grid_locate_y'], 'img_class': sample['img_class']}

##testing##
dataTransform = transforms.Compose([Rescale(200),RandomCrop(190),give_value(), Normalize(), ToTensor()])
dataset = yoloDataset(rootDirImg = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/github/dataset_colab/images/mixed',
rootDirCoord = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/github/dataset_colab/texts/mixed',transform=dataTransform)

num_of_pics = 1

'''for i in range(0, num_of_pics):
    idx = np.random.randint(0, len(dataset))
    #idx = 0
    sample = dataset[idx]
    #print(sample)
    #print('img name:',sample['img_name'])
    print("sample['grid_locate_x']", sample['grid_locate_x'])
    print('tensor:', sample['coord'][sample['grid_locate_y']][sample['grid_locate_x']])
    print('img_class',sample['img_class'])'''
    '''
    sample['grid_locate_x'] 4
    tensor: [-2.     -2.     -2.     -2.     -2.     -2.     -2.     -1.98   -1.98
    -2.     -1.69   -1.97   -1.7765 -1.855 ]
    img_class 1

    '''
    #showCenter(img = sample['image'], xmin = sample['coord'][0],xmax = sample['coord'][1],ymin = sample['coord'][2],ymax = sample['coord'][3])

'''showCenter(Imgpath = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-27.jpg',TxtrootDirCoordPath = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-27.txt')'''