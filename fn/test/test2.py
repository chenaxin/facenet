# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import os
import copy
from embeddings1 import load_model
sys.path.append('../align/')
from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames
import config
import cv2
import h5py
import utils1
#识别人脸阈值
THRED=0.0022
#带人脸对齐

# In[2]:

def main():
    #读取对比图片的embeddings和class_name
    f=h5py.File('../pictures/embeddings.h5','r')
    class_arr=f['class_name'][:]
    class_arr=[k.decode() for k in class_arr]
    emb_arr=f['embeddings'][:]
    file_paths = get_model_filenames('../align/save_model/all_in_one/')
    #mtcnn_detector=load_align()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model('../model/')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            
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
            t1=cv2.getTickCount()
            img=cv2.imread("D:\\people.jpg")
                #检测人脸
            rectangles, points = detect_face(img, 20,
                                                pnet_fun, rnet_fun, onet_fun,
                                                [0.5, 0.6, 0.7], 0.7)
            scaled_gray=[]
            if len(rectangles)!=0:
                for i in range(len(rectangles)):
                    if rectangles[i] is not None:
                        crop_img = img[int(rectangles[i][1]):int(rectangles[i][3]), int(rectangles[i][0]):int(rectangles[i][2])]
                        crop_img=utils1.align_face(crop_img, points[:,i])
                        if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                            continue
                        if crop_img is None:
                            continue
                        #print(crop_img.shape)
                        scaled_gray.extend([cv2.resize(cv2.cvtColor(crop_img,cv2.COLOR_RGB2GRAY),(160,160))])
                        cv2.rectangle(img, (int(rectangles[i][0]), 
                                int(rectangles[i][1])),                                                    
                                (int(rectangles[i][2]), int(rectangles[i][3])),
                                (255, 0, 0), 1)
                scaled_gray=np.float64(np.array(scaled_gray))
                scaled_gray=scaled_gray.reshape(scaled_gray.shape[0],160,160,1)
                feed_dict = { images_placeholder: scaled_gray, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
                embs = sess.run(embeddings, feed_dict=feed_dict)
                face_num=embs.shape[0]
                face_class=['Others']*face_num
                for i in range(face_num):
                    diff=np.mean(np.square(embs[i]-emb_arr),axis=1)
                    min_diff=min(diff)
                    print(min_diff)
                    if min_diff<THRED:
                        index=np.argmin(diff)
                        face_class[i]=class_arr[index]
              
                    cv2.putText(img, '{}'.format(face_class[i]), 
                            (int(rectangles[i,0]),int(rectangles[i,1])), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0, 255, 0), 2)
                t2=cv2.getTickCount()
                #t=(t2-t1)/cv2.getTickFrequency()
                fps=cv2.getTickFrequency()/(t2-t1)
                cv2.putText(img, '{}'.format(fps), 
                            (20,20), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0, 255, 0), 2)              
            cv2.imshow("test", img)
            cv2.waitKey(0)





if __name__=='__main__':
    main()