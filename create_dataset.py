import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
def create_dataset_function():
    training_data=[]
    DATADIR = "D:\geordi\ecole\Automn2022\SYS843\pycharmprojects"
    CATEGORIES = ["images_atrial_premature","images_left_bundle","images_normal","images_paced_beat","images_right_bundle","images_ventricular_escape","images_ventricular_premature"]
    IM_SIZE = 128
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        print(path)
        for img in os.listdir(path):
            try:
                print(os.path.join(path, img))
                im_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
                training_data.append([im_array,class_num])
            except Exception as e:
                pass


