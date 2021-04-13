
# coding: utf-8

# In[1]:

import numpy as np
import math
import sys
import time
import tensorflow as tf
import cv2
#from operator import itemgetter
from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames
import matplotlib.pyplot as plt
import config
import os
import random
from tqdm import tqdm
from utils1 import *

# In[3]:

def main():
    file_paths = get_model_filenames('./save_model/all_in_one/')
    with tf.Graph().as_default():
        with tf.Session() as sess:
            #恢复模型
            saver = tf.train.import_meta_graph(file_paths[0])
            saver.restore(sess, file_paths[1])
            def pnet_fun(img): return sess.run(
                ('softmax/Reshape_1:0',
                'pnet/conv4-2/BiasAdd:0'),
                feed_dict={
                    'Placeholder:0': img})

            def rnet_fun(img): return sess.run(
                ('softmax_1/softmax:0',
                'rnet/conv5-2/rnet/conv5-2:0'),
                feed_dict={
                    'Placeholder_1:0': img})

            def onet_fun(img): return sess.run(
                ('softmax_2/softmax:0',
                'onet/conv6-2/onet/conv6-2:0',
                'onet/conv6-3/onet/conv6-3:0'),
                feed_dict={
                    'Placeholder_2:0': img})
            thresh=config.thresh
            min_face_size=config.min_face
            stride=config.stride
            detectors=[None,None,None]
            # 模型放置位置
            batch_size=config.batches
            out_path=config.out_path
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            #选用图片
            path=config.test_dir
            #获取图片类别和路径
            dataset = get_dataset(path)
            random.shuffle(dataset)
            # In[4]:
            bounding_boxes_filename = os.path.join(out_path, 'bounding_boxes.txt')

            with open(bounding_boxes_filename, "w") as text_file:
                for cls in tqdm(dataset):
                    output_class_dir = os.path.join(out_path, cls.name)
                    if not os.path.exists(output_class_dir):
                        os.makedirs(output_class_dir)
                        random.shuffle(cls.image_paths)
                    for image_path in cls.image_paths:
                        #得到图片名字如001
                        filename = os.path.splitext(os.path.split(image_path)[1])[0]
                        output_filename = os.path.join(output_class_dir, filename+'.jpg')
                        if not os.path.exists(output_filename):
                            try:
                                img = cv2.imread(image_path)
                            except (IOError, ValueError, IndexError) as e:
                                errorMessage = '{}: {}'.format(image_path, e)
                                print(errorMessage)
                            else:
                                if img.ndim<3:
                                    print('图片不对劲 "%s"' % image_path)
                                    text_file.write('%s\n' % (output_filename))
                                    continue
                                
                                img = img[:,:,0:3]
                                #通过mtcnn获取人脸框
                                try:
                                    boxes_c, points = detect_face(img, min_face_size,
                                                        pnet_fun, rnet_fun, onet_fun,
                                                        thresh, 0.7)
                                    #boxes_c,_=mtcnn_detector.detect(img)
                                except:
                                    print('识别不出图像:{}'.format(image_path))
                                    continue
                                #人脸框数量
                                num_box=boxes_c.shape[0]
                                if num_box>0:
                                    det=boxes_c[:,:4]
                                    det_arr=[]
                                    p_arr=[]
                                    img_size=np.asarray(img.shape)[:2]
                                    if num_box>1:
                                        if config.detect_multiple_faces:
                                            for i in range(num_box):
                                                det_arr.append(np.squeeze(det[i]))
                                                p_arr.append(np.squeeze(points[:,i]))
                                        else:
                                            #如果保留一张脸，但存在多张，只保留置信度最大的
                                            score=boxes_c[:,4]
                                            index=np.argmax(score)
                                            det_arr.append(det[index,:])
                                            p_arr.append(points[:,index])
                                    else:
                                        det_arr.append(np.squeeze(det))
                                        p_arr.append(np.squeeze(points))
                                    for i,det in enumerate(det_arr):
                                        det=np.squeeze(det)
                                        bb=[int(max(det[0],0)), int(max(det[1],0)), int(min(det[2],img_size[1])), int(min(det[3],img_size[0]))]
                                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                        cropped=align_face(cropped, p_arr[i])
                                        try:
                                            scaled =cv2.resize(cropped,(config.image_size, config.image_size),interpolation=cv2.INTER_LINEAR)
                                        except:
                                            print('识别不出的图像：{}，box的大小{},{},{},{}'.format(image_path,bb[0],bb[1],bb[2],bb[3]))
                                            continue
                                        filename_base, file_extension = os.path.splitext(output_filename)
                                        if config.detect_multiple_faces:
                                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                        else:
                                            output_filename_n = "{}{}".format(filename_base, file_extension)
                                            #scaled=cv2.cvtColor(scaled,cv2.COLOR_BGR2GRAY)
                                            cv2.imwrite(output_filename_n,scaled)
                                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                                else:
                                    print('图像不能对齐 "%s"' % image_path)
                                    text_file.write('%s\n' % (output_filename))

if __name__=='__main__':
    main()


# In[2]:
