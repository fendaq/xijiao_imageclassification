import cv2
import os
'''
make train use 
'''

train_path = 'D:/data/datasets_fusai/datasets/train.txt'
train_dir = 'D:/data/datasets_fusai/datasets/train/'

f = open('train_use2.txt','w')
lists = list()
with open(train_path) as rf:
    lines = rf.readlines()
    for line in lines:
        contents = line.split(' ')
        img_name = contents[0]
        img_path = os.path.join(train_dir, img_name)
        img_label = contents[1]
        xmin = contents[2]
        ymin = contents[3]
        xmax = contents[4]
        ymax = contents[5]
        lists.append(img_path+' '+xmin+' '+ymin+' '+xmax+' '+ymax.strip()+' '+img_label.strip()+'\n')

f.writelines(lists)
f.close()