import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import re

def main(txt_folder = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/', img_folder = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/'):
    txt_dog_fold = txt_folder
    txt_dog_name_arr = os.listdir(txt_dog_fold)
    #print(name_txt_dog)
    img_dog_name_arr = []
    for i in np.arange(len(txt_dog_name_arr)):
        #print(txt_dog_name_arr[i])
        convt = txt_dog_name_arr[i].replace('.txt','.jpg')
        img_dog_name_arr.append(convt)
        dog_name = img_folder + convt
        frame = cv2.imread(dog_name,1)
        frame = cv2.resize(frame, (300,300), interpolation = cv2.INTER_AREA)
        return frame
        #print(frame.shape)
        '''cv2.imshow('dog', frame)
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
    print(img_dog_name_arr)'''

def load_img(txt_file = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-1.txt', img_file = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-1.jpg'):
    #txt_dog_name_arr = os.listdir(txt_dog_fold)
    #print(name_txt_dog)
    #img_dog_name_arr = []
    #print(txt_dog_name_arr[i])
    frame = cv2.imread(img_file,1)
    frame = cv2.resize(frame, (300,300), interpolation = cv2.INTER_AREA)
    '''plt.imshow(frame,cmap='gray')
    plt.show()'''
    return frame
    #print(frame.shape)
    '''cv2.imshow('dog', frame)
    if cv2.waitKey(0) == ord('q'):
        break
    cv2.destroyAllWindows()'''

#load_img()

def readCoord(path = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-1.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = re.split(r'\s', lines[1])
        lines = list(map(int, lines))
        return lines
        #print(lines)
        #[40, 234, 35, 213]

#readCoord()

def findCentroid(path = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-1.txt'):
    coord = readCoord(path = path)
    #xmin, ymin = coord[0]/2, coord[2]/2
    xcenter , ycenter = (coord[1]+coord[0])/2 , (coord[3]+coord[2])/2
    #print(type(xmax))
    centroid_arr = np.array([xcenter, ycenter])
    #centroid_arr.reshape(1,-1)
    #centroid_arr = np.squeeze(centroid_arr)
    #print(centroid_arr)
    #[ 20.  117.   17.5 106.5]
    return centroid_arr
    #print(coord)
    #[40, 234, 35, 213]

#findCentroid()

def showCenter(Imgpath = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-1.jpg',TxtBoundingCoordPath = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/texts/dogs_txts/dog-1.txt'):
    #gridFrame = grid(path = path)
    frame = cv2.imread(Imgpath,0)
    centroidArr = findCentroid(path = TxtBoundingCoordPath)
    frame = cv2.circle(frame, (int(centroidArr[0]), int(centroidArr[1])), 2, (0,0,255), -1)
    coord = readCoord()
    frame = cv2.rectangle(frame, (coord[0], coord[2]), (coord[1], coord[3]), (0,0,255), 2)
    #print(centroidArr)
    #[137. 124.]
    #print(coord)
    #[40, 234, 35, 213]
    plt.imshow(frame, cmap='gray')
    plt.show()

#showCenter()

def grid(path = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/dogs_imgs/dog-1.jpg',height = 15, width = 15):
    tx = 0
    ty = 0
    bx = 15
    by = 15
    frame = load_img(img_file = path)
    frame_copy = np.copy(frame)
    for j in range(0,300,height):
        for i in range(0,300,width):
            tx = tx + i
            ty = ty + j
            bx = bx + i
            by = by + j
            frame_copy = cv2.rectangle(frame_copy,(tx,ty),(bx,by), (0,255,0), 2)
            tx = 0
            ty = 0
            bx = 15
            by = 15
        if j == 300:
            break
    #plt.imshow(frame_copy, cmap='gray')
    #plt.show()
    return frame_copy
#grid()