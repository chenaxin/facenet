# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import os
import copy
sys.path.append('../align/')
from src.mtcnn import PNet, RNet, ONet
from tools import detect_face, get_model_filenames
#from utils import *
import utils1
import config
import cv2
import h5py


# In[2]:

def main():
    path='../pictures/embeddings.h5'
    if os.path.exists(path):
        print('生成完了别再瞎费劲了！！！')
        return

    with tf.Graph().as_default():
        with tf.Session() as sess:
            file_paths = get_model_filenames('../align/save_model/all_in_one/')
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
            #选用图片
            #获取图片类别和路径
            path='../pictures/'
            img_paths=os.listdir(path)
            class_names=[a.split('.')[0] for a in img_paths]
            img_paths=[os.path.join(path,p) for p in img_paths]
            scaled_arr=[]
            class_names_arr=[]
            
            for image_path,class_name in zip(img_paths,class_names):
                
                img = cv2.imread(image_path)
        #         cv2.imshow('',img)
        #         cv2.waitKey(0)
                try:
                    rectangles, points = detect_face(img, 20,
                                        pnet_fun, rnet_fun, onet_fun,
                                        [0.5, 0.6, 0.7], 0.7)
                except:
                    print('识别不出图像:{}'.format(image_path))
                    continue
                #人脸框数量
                num_box=rectangles.shape[0]
                if num_box>0:
                    det=rectangles[:,:4]
                    det_arr=[]
                    p_arr=[]
                    img_size=np.asarray(img.shape)[:2]
                    if num_box>1:
                        #如果保留一张脸，但存在多张，只保留置信度最大的
                        score=rectangles[:,4]
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
                        cropped=utils1.align_face(cropped, p_arr[i][:])
                        scaled =cv2.resize(cropped,(160, 160),interpolation=cv2.INTER_LINEAR)-127.5/128.0
                        scaled_arr.append(scaled)
                        class_names_arr.append(class_name)

                else:
                    print('图像不能对齐 "%s"' % image_path)
            scaled_arr=np.asarray(scaled_arr)
            class_names_arr=np.asarray(class_names_arr)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            img_gray=[]
            scaled_arr=np.uint8(scaled_arr)
            for i in range(scaled_arr.shape[0]):
                img_gray.extend([cv2.cvtColor(scaled_arr[i],cv2.COLOR_RGB2GRAY)])
                #img_gray.extend([TV12(cv2.cvtColor(scaled_arr[i],cv2.COLOR_RGB2GRAY),0.05,100)])
            img_gray=np.float64(np.array(img_gray)).reshape(scaled_arr.shape[0],160,160,1)
            load_model('../model/')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')

            # 前向传播计算embeddings
            feed_dict = { images_placeholder: img_gray, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
            embs = sess.run(embeddings, feed_dict=feed_dict)
    f=h5py.File('../pictures/embeddings.h5','w')
    class_names_arr=[i.encode() for i in class_names_arr]
    f.create_dataset('class_name',data=class_names_arr)
    f.create_dataset('embeddings',data=embs)
    f.close()


# In[3]:


# In[4]:


def load_model(model_dir,input_map=None):
    '''重载模型'''
    
    ckpt = tf.train.get_checkpoint_state(model_dir)                         
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   
    saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)


# In[ ]:


if __name__=='__main__':
    main()