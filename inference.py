import torch
import numpy as np
import cv2
from infer_utils import infer_image,load_model
from glob import glob
from centernet import CenterNet_Resnet50
import os
from CenternetPlus.centernet_plus import CenterNetPlus
import time 

f = open("classes.txt","r").readlines()
classes = []
for i in f:
    classes.append(i.strip('\n'))

print(classes)


folder = "/home/rivian/Desktop/Datasets/coco_mini_train"
folder = os.path.join(folder,"val_images") #place to val

model_path = "best_epoch_weights_cplus_r18_cocomini_0.162map.pth"
device = "cuda"


CPlus = True
if CPlus == False:
    pretrained = True
    conf = 0.2
    model = CenterNet_Resnet50(len(classes), pretrained = pretrained)
    if model_path != "":
        model = load_model(model,model_path)

else:
    conf = 0.2
    model = CenterNetPlus(True,len(classes),'r18')
    if model_path != "":
        model = load_model(model,model_path)

model.cuda()

video = True
half = True 

if half:
    model.half()

video_path = "/home/rivian/Desktop/bottle-detection.mp4"   #17_2022-11-09-14.26.56_derpet-converted.mp4"
#video_path = 0
if video:
    cap = cv2.VideoCapture(video_path)
    while 1:
        ret,img = cap.read()
        image = infer_image(model,img,classes,conf,half,input_shape=(512,512))
        cv2.imshow("img",image)
        ch = cv2.waitKey(1)
        if ch == ord("q"): break


else:
    files = glob(folder+"/*.jpg")
    for i in files:
        image = infer_image(model,i,classes,conf,half)
        cv2.imshow("img",image)
        ch = cv2.waitKey(0)
        if ch == ord("q"): break
