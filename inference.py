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


folder = "/home/rivian/Desktop/Datasets/derpet_v4_label_tf" #"/home/rivian/Desktop/Datasets/coco_mini_train"
folder = os.path.join(folder,"val_images") #place to val

model_path = "best_epoch_weights.pth" #"pretrained-hardnet.pth"
device = "cuda"
model_type = "shufflenet"


if model_type == "shufflenet":
    from lib.core.model.centernet import CenterNet
    conf = 0.25
    model = CenterNet(nc=len(classes))
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "resnet50":
    pretrained = True
    conf = 0.2
    model = CenterNet_Resnet50(len(classes), pretrained = pretrained)
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "CenterNetPlus":
    conf = 0.4
    model = CenterNetPlus(True,len(classes),'r18')
    if model_path != "":
        model = load_model(model,model_path)


if model_type == "detr":
    from detr.detrmodel import *
    hidden_dim = 256
    nheads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    model = DETR(len(classes),hidden_dim,nheads,num_encoder_layers,num_decoder_layers)
    conf = 0.7
    #model_path = ""
    device = "cuda"
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "timmodel":
    from timmodel.timmodel import *
    conf = 0.7
    model = centernet(classes)
    model_path = ""
    device = "cuda"
    if model_path != "":
        model = load_model(model,model_path)

if model_type == "DLA":
    from dlamodel import * 
    conf = 0.1
    device = "cuda"
    model = get_pose_net("34",{"hm":len(classes),"wh":2,"offset":2})
    if model_path != "":
        model = load_model(model,model_path)   

if model_type == "hardnet":
    from hardnet import get_pose_net
    device = "cuda"
    conf = 0.2
    model = get_pose_net(85,{"hm":len(classes),"offset":2,"wh":2})
    if model_path != "":
        model = load_model(model,model_path)

model.cuda()

#model = torch.compile(model) #experimental

model.eval()
video = True
half = False 
cpu = True 
trace = True 
openvino_exp = True 

if cpu:
    model.cpu()
    device = torch.device("cpu")

if half:
    model.half()

if trace:
    input_height = 320
    input_width = 320
    dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
    print("Start Tracing")
    model = torch.jit.trace(model, dummy_input)
    print("End Tracing")

if openvino_exp:
    import openvino as ov
    model =  ov.compile_model(ov.convert_model(model, example_input=dummy_input))




video_path = "G:/2_18.00.00_novis_output.avi"
#video_path = 0
if video:
    cap = cv2.VideoCapture(video_path)
    while 1:
        ret,img = cap.read()
        image,annos = infer_image(model,img,classes,conf,half,input_shape=(320,320),cpu=cpu,openvino_exp=openvino_exp)
        #print(annos)
        cv2.imshow("img",image)
        ch = cv2.waitKey(1)
        if ch == ord("q"): break


else:
    files = glob(folder+"/*.jpg") + glob(folder+"/*.png")
    for i in files:
        image,annos = infer_image(model,i,classes,conf,half,input_shape=(320,320),cpu=cpu,openvino_exp=openvino_exp)
        cv2.imshow("img",image)
        ch = cv2.waitKey(0)
        if ch == ord("q"): break
