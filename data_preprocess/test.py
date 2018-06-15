import cv2
import os


train_path = 'D:/data/datasets_fusai/datasets/train.txt'
train_dir = 'D:/data/datasets_fusai/datasets/train/'
with open(train_path) as rf:
    lines = rf.readlines()
    for line in lines:
        contents = line.split(' ')
        img_name = contents[0]
        img_path = os.path.join(train_dir, img_name)
        np_img = cv2.imread(img_path)
        img_label = contents[1]
        xmin = contents[2]
        ymin = contents[3]
        xmax = contents[4]
        ymax = contents[5]
        print(ymax)
        cv2.rectangle(np_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
        cv2.putText(np_img, img_label, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
        cv2.imwrite('D:/data/datasets_fusai/datasets/show/'+img_name, np_img)