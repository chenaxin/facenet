import sys

from operator import itemgetter

import os
from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt

import numpy as np

import math

#-----------------------------#

#   计算原始输入图像

#   每一次缩放的比例

#-----------------------------#

def calculateScales(img):

    pr_scale = 1.0

    h,w,_ = img.shape

    

    #--------------------------------------------#

    #   将最大的图像大小进行一个固定

    #   如果图像的短边大于500，则将短边固定为500

    #   如果图像的长边小于500，则将长边固定为500

    #--------------------------------------------#

    if min(w,h)>500:

        pr_scale = 500.0/min(h,w)

        w = int(w*pr_scale)

        h = int(h*pr_scale)

    elif max(w,h)<500:

        pr_scale = 500.0/max(h,w)

        w = int(w*pr_scale)

        h = int(h*pr_scale)



    #------------------------------------------------#

    #   建立图像金字塔的scales，防止图像的宽高小于12

    #------------------------------------------------#

    scales = []

    factor = 0.709

    factor_count = 0

    minl = min(h,w)

    while minl >= 12:

        scales.append(pr_scale*pow(factor, factor_count))

        minl *= factor

        factor_count += 1

    return scales



#-----------------------------#

#   将长方形调整为正方形

#-----------------------------#

def rect2square(rectangles):

    w = rectangles[:,2] - rectangles[:,0]

    h = rectangles[:,3] - rectangles[:,1]

    l = np.maximum(w,h).T

    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5

    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 

    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 

    return rectangles



#-------------------------------------#

#   非极大抑制

#-------------------------------------#

def NMS(rectangles,threshold):

    if len(rectangles)==0:

        return rectangles

    boxes = np.array(rectangles)

    x1 = boxes[:,0]

    y1 = boxes[:,1]

    x2 = boxes[:,2]

    y2 = boxes[:,3]

    s  = boxes[:,4]

    area = np.multiply(x2-x1+1, y2-y1+1)

    I = np.array(s.argsort())

    pick = []

    while len(I)>0:

        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others

        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])

        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])

        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)

        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)

        pick.append(I[-1])

        I = I[np.where(o<=threshold)[0]]

    result_rectangle = boxes[pick].tolist()

    return result_rectangle



#-------------------------------------#

#   对pnet处理后的结果进行处理

#   为了方便理解，我将代码进行了重构

#   具体代码与视频有较大区别

#-------------------------------------#

def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):

    #-------------------------------------#

    #   计算特征点之间的步长

    #-------------------------------------#

    stride = 0

    if out_side != 1:

        stride = float(2*out_side-1)/(out_side-1)



    #-------------------------------------#

    #   获得满足得分门限的特征点的坐标

    #-------------------------------------#

    (y,x) = np.where(cls_prob >= threshold)

    

    #-----------------------------------------#

    #   获得满足得分门限的特征点得分

    #   最终获得的score的shape为：[num_box, 1]

    #-------------------------------------------#

    score = np.expand_dims(cls_prob[y, x], -1)



    #-------------------------------------------------------#

    #   将对应的特征点的坐标转换成位于原图上的先验框的坐标

    #   利用回归网络的预测结果对先验框的左上角与右下角进行调整

    #   获得对应的粗略预测框

    #   最终获得的boundingbox的shape为：[num_box, 4]

    #-------------------------------------------------------#

    boundingbox = np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis = -1)

    top_left = np.fix(stride * boundingbox + 0)

    bottom_right = np.fix(stride * boundingbox + 11)

    boundingbox = np.concatenate((top_left,bottom_right), axis = -1)

    boundingbox = (boundingbox + roi[y, x] * 12.0) * scale

    

    #-------------------------------------------------------#

    #   将预测框和得分进行堆叠，并转换成正方形

    #   最终获得的rectangles的shape为：[num_box, 5]

    #-------------------------------------------------------#

    rectangles = np.concatenate((boundingbox, score), axis = -1)

    rectangles = rect2square(rectangles)

    

    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)

    return rectangles

    

#-------------------------------------#

#   对Rnet处理后的结果进行处理

#   为了方便理解，我将代码进行了重构

#   具体代码与视频有较大区别

#-------------------------------------#

def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):

    #-------------------------------------#

    #   利用得分进行筛选

    #-------------------------------------#

    pick = cls_prob[:, 1] >= threshold



    score  = cls_prob[pick, 1:2]

    rectangles = rectangles[pick, :4]

    roi = roi[pick, :]



    #-------------------------------------------------------#

    #   利用Rnet网络的预测结果对粗略预测框进行调整

    #   最终获得的rectangles的shape为：[num_box, 4]

    #-------------------------------------------------------#

    w   = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)

    h   = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)

    rectangles[:, [0,2]]  = rectangles[:, [0,2]] + roi[:, [0,2]] * w

    rectangles[:, [1,3]]  = rectangles[:, [1,3]] + roi[:, [1,3]] * w



    #-------------------------------------------------------#

    #   将预测框和得分进行堆叠，并转换成正方形

    #   最终获得的rectangles的shape为：[num_box, 5]

    #-------------------------------------------------------#

    rectangles = np.concatenate((rectangles,score), axis=-1)

    rectangles = rect2square(rectangles)



    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)

    return np.array(NMS(rectangles, 0.7))



#-------------------------------------#

#   对onet处理后的结果进行处理

#   为了方便理解，我将代码进行了重构

#   具体代码与视频有较大区别

#-------------------------------------#

def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):

    #-------------------------------------#

    #   利用得分进行筛选

    #-------------------------------------#

    pick = cls_prob[:, 1] >= threshold



    score  = cls_prob[pick, 1:2]

    rectangles = rectangles[pick, :4]

    pts = pts[pick, :]

    roi = roi[pick, :]



    w   = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)

    h   = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)

    #-------------------------------------------------------#

    #   利用Onet网络的预测结果对预测框进行调整

    #   通过解码获得人脸关键点与预测框的坐标

    #   最终获得的face_marks的shape为：[num_box, 10]

    #   最终获得的rectangles的shape为：[num_box, 4]

    #-------------------------------------------------------#

    face_marks = np.zeros_like(pts)

    face_marks[:, [0,2,4,6,8]] = w * pts[:, [0,1,2,3,4]] + rectangles[:, 0:1]

    face_marks[:, [1,3,5,7,9]] = h * pts[:, [5,6,7,8,9]] + rectangles[:, 1:2]

    rectangles[:, [0,2]]  = rectangles[:, [0,2]] + roi[:, [0,2]] * w

    rectangles[:, [1,3]]  = rectangles[:, [1,3]] + roi[:, [1,3]] * w

    #-------------------------------------------------------#

    #   将预测框和得分进行堆叠

    #   最终获得的rectangles的shape为：[num_box, 15]

    #-------------------------------------------------------#

    rectangles = np.concatenate((rectangles,score,face_marks),axis=-1)



    rectangles[:, [1,3]] = np.clip(rectangles[:, [1,3]], 0, height)

    rectangles[:, [0,2]] = np.clip(rectangles[:, [0,2]], 0, width)

    return np.array(NMS(rectangles,0.3))

#人脸对齐
def align_face(image_array, landmarks):
    x=landmarks[0]-landmarks[2]
    y=landmarks[1]-landmarks[3]
    if x==0:
        angle=0
    else:
        angle=math.atan(y/x)*180/math.pi
    center=(image_array.shape[1]//2,image_array.shape[0]//2)
    RotationMatrix=cv2.getRotationMatrix2D(center,angle,1)
    rotated_img=cv2.warpAffine(image_array,RotationMatrix,(image_array.shape[1],image_array.shape[0]))

    return rotated_img




def IOU(box,boxes):
    '''裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    '''
    #box面积
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    #boxes面积,[n,]
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    #重叠部分左上右下坐标
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])

    #重叠部分长宽
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    #重叠部分面积
    inter=w*h
    return inter/(box_area+area-inter+1e-10)


# In[3]:

def read_annotation(base_dir, label_path):
    '''读取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')

        one_image_bboxes = []
        for i in range(int(nums)):

            bb_info = labelfile.readline().strip('\n').split(' ')
            #人脸框
            face_box = [float(bb_info[i]) for i in range(4)]

            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]

            one_image_bboxes.append([xmin, ymin, xmax, ymax])

        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    return data
def convert_to_square(box):
    '''将box转换成更大的正方形
    参数：
      box：预测的box,[n,5]
    返回值：
      调整后的正方形box，[n,5]
    '''
    square_box=box.copy()
    h=box[:,3]-box[:,1]+1
    w=box[:,2]-box[:,0]+1
    #找寻正方形最大边长
    max_side=np.maximum(w,h)

    square_box[:,0]=box[:,0]+w*0.5-max_side*0.5
    square_box[:,1]=box[:,1]+h*0.5-max_side*0.5
    square_box[:,2]=square_box[:,0]+max_side-1
    square_box[:,3]=square_box[:,1]+max_side-1
    return square_box
class ImageClass():
    '''获取图片类别和路径'''
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):
    dataset = []
    classes = [path for path in os.listdir(paths) if os.path.isdir(os.path.join(paths, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in tqdm(range(nrof_classes)):
        class_name = classes[i]
        facedir = os.path.join(paths, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def split_dataset(dataset,split_ratio,min_nrof_images_per_class):
    '''拆分训练和验证集
    参数：
      dataset:有get_dataset生成的数据集
      split_ratio:留取验证集的比例
      min_nrof_images_per_class：一个类别中最少含有的图片数量，过少舍弃
    返回值：
      train_set,test_set:还有图片类别和路径的训练验证集
    '''
    train_set=[]
    test_set=[]
    for cls in dataset:
        paths=cls.image_paths
        np.random.shuffle(paths)
        #某一种类图片个数
        nrof_images_in_class=len(paths)
        #留取训练的比例
        split=int(math.floor(nrof_images_in_class*(1-split_ratio)))
        if split==nrof_images_in_class:
            split=nrof_images_in_class-1
        if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
            train_set.append(ImageClass(cls.name,paths[:split]))
            test_set.append(ImageClass(cls.name,paths[split:]))
    return train_set,test_set

def get_image_paths_and_labels(dataset):
    '''获取所有图像地址和类别'''
    image_paths_flat=[]
    labels_flat=[]
    for i in range(len(dataset)):
        image_paths_flat+=dataset[i].image_paths
        labels_flat+=[i]*len(dataset[i].image_paths)
    return image_paths_flat,labels_flat

def create_input_pipeline(input_queue,image_size,nrof_preprocess_threads,bath_size_placeholder):
    '''由输入队列返回图片和label的batch组合
    参数：
      input_queue:输入队列
      image_size:图片尺寸
      nrof_preprocess_threads:线程数
      batch_size_placeholder:batch_size的placeholder
    返回值：
      image_batch,label_batch:图片和label的batch组合
    '''
    image_and_labels_list=[]
    for _ in range(nrof_preprocess_threads):
        filenames,label=input_queue.dequeue()
        images=[]
        for filename in tf.unstack(filenames):
            file_contents=tf.read_file(filename)
            image=tf.image.decode_image(file_contents,1)
            #随机翻转图像
            image=tf.cond(tf.constant(np.random.uniform()>0.8),
                          lambda:tf.py_func(random_rotate_image,[image],tf.uint8),
                          lambda:tf.identity(image))
            #随机裁剪图像
            image=tf.cond(tf.constant(np.random.uniform()>0.5),
                          lambda:tf.random_crop(image,image_size+(1,)),
                          lambda:tf.image.resize_image_with_crop_or_pad(image,image_size[0],image_size[1]))
            #随机左右翻转图像
            image=tf.cond(tf.constant(np.random.uniform()>0.7),
                          lambda:tf.image.random_flip_left_right(image),
                          lambda:tf.identity(image))
            #图像归一到[-1,1]内
            image=(tf.cast(image,tf.float32)-127.5)/128.0
            image.set_shape(image_size+(1,))
            images.append(image)
        image_and_labels_list.append([images,label])
    image_batch,label_batch=tf.train.batch_join(image_and_labels_list,
                                  batch_size=bath_size_placeholder,
                                  shapes=[image_size+(1,),()],
                                  enqueue_many=True,
                                  capacity=4*nrof_preprocess_threads*100,
                                  allow_smaller_final_batch=True)
    return image_batch,label_batch

def random_rotate_image(image):
    '''随机翻转图片'''
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')
