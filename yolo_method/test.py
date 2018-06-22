import sys
import os
sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19","20",
"21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39","40",
"41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59","60"]


def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:60]
  prob = np.max(p_classes)
  C = predicts[0, :, :, 60:62]
  coordinate = predicts[0, :, :, 62:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 60))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]

  index = np.argmax(P)
  print(index)

  index = np.unravel_index(index, P.shape)
  print(index)

  class_num = index[3]
  print(class_num)
  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num, prob

common_params = {'image_size': 448, 'num_classes': 60, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        with open('E:/data/datasets/test_res.csv','a') as wf:
            with open('E:/data/datasets/test.txt') as rf:
                for line in rf.readlines():
                    img_path = line.strip("\n").split(" ")[0]
                    test_dir = "E:/data/datasets/train/"
                    img = cv2.imread(os.path.join(test_dir, img_path))
                    img_h = img.shape[0]
                    img_w = img.shape[1]

                    width_rate = img_w / 448 * 1.0 
                    height_rate = img_w / 448 * 1.0 

                    resized_img = cv2.resize(img, (448, 448))
                    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


                    np_img = np_img.astype(np.float32)

                    np_img = np_img / 255.0 * 2 - 1
                    np_img = np.reshape(np_img, (1, 448, 448, 3))
                    print(np_img.shape)
                    saver = tf.train.Saver(net.trainable_collection)

                    saver.restore(sess, 'D:/Git/xijiao_imageclassification/yolo_method/tools/model/model.ckpt-300000')

                    np_predict = sess.run(predicts, feed_dict={image: np_img})

                    xmin, ymin, xmax, ymax, class_num, prob = process_predicts(np_predict)
                    class_name = classes_name[class_num]
                    wf.write("%s %d %f %d %d %d %d\n"%(img_path, int(class_name), prob, int(xmin*width_rate), int(ymin*height_rate), int(xmax*width_rate), int(ymax*height_rate)))
except tf.errors.OutOfRangeError:
    print('finished')
finally:
    coord.request_stop()
coord.join(threads)
"""
cv2.rectangle(img, (int(xmin*width_rate), int(ymin*height_rate)), (int(xmax*width_rate), int(ymax*height_rate)), (0, 0, 255))
cv2.putText(img, class_name, (int(xmin*width_rate), int(ymin*height_rate)), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out.jpg', img)
"""
sess.close()