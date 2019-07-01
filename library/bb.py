import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import os

#global constants
img = None
tl_list = []
br_list = []
obj_list = []
img_save_list = []
counter = 0
#linux
#img_folder = 'E:/for development purpose only/python/my_yolo/dataset'
#annotation = 'E:/for development purpose only/python/my_yolo/dataset/annotations'

PATH_IMAGE = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/test/img'
path_img_list = os.listdir(PATH_IMAGE)

#windows in spyder
img_folder = '/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/test/img'
annotation = 'annotations'

obj = 'license'

def line_select_callback(clk, rls):
    global tl_list
    global br_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    obj_list.append(obj)
    print(obj_list)


def on_key_press(event):
    global obj_list
    global tl_list
    global br_list
    global img
    if event.key == 'q':
        print(tl_list, br_list)
        #img_save = img_save.replace('jpg', 'txt')
        img_name = path_img_list[counter-1]
        img_name = img_name.replace('jpg', 'txt')
        with open('/media/mrugank/626CB0316CB00239/for development purpose only/python/computer_vision/part_1_mod_1_lsn_2/yolo/dataset/test/txt/'+img_name, 'w') as save_txt:
            save_txt.flush()
            save_txt.write('1'+'\n')
            save_txt.write('{} {} {} {}'.format(tl_list[0][0], br_list[0][0],tl_list[0][1], br_list[0][1]))
        tl_list = []
        br_list = []
        img = None
        img_save = None
        obj_list = []
        plt.close()

def toggle_selector(event):
    toggle_selector.RS.set_active(True)

if __name__ == '__main__':
    for i, img_file in enumerate(os.scandir(img_folder)):
        #img_save = img_file
        img = img_file
        img_save_list.append(img_file)
        counter += 1
        print('image',img)
        fig , ax = plt.subplots(1)
        img = cv2.imread(img_file.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        bbox =  plt.connect('key_press_event', line_select_callback)
        key = plt.connect('key_press_event', on_key_press)
        plt.show()
