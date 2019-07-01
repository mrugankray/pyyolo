import os

dataset = os.listdir('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/person')

print(dataset)

j = 0 

for i in range(0, len(dataset)):
    source = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/images/person/' + dataset[i]
    os.rename(source, 'dataset/images/person/person{}.jpg'.format(j))
    j = j+1