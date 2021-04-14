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

with tf.Session()as sess:
    saver=tf.train.import_meta_graph("C:\\Users\\bit\\Desktop\\fn\\model\\model.ckpt-279261.meta")
    saver.restore(sess, "C:\\Users\\bit\\Desktop\\fn\\model\\model.ckpt-279261")
    var=tf.global_variables()
    for i in range(len(var)):
        print(var[i].value())