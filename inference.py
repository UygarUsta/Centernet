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


folder = r"C:\Users\PC\Pictures\_" #r"G:\COCO\val2017" #"E:/derpetv5_xml" #"/home/rivian/Desktop/Datasets/derpet_v4_label_tf" #"/home/rivian/Desktop/Datasets/coco_mini_train"
#folder = os.path.join(folder,"val_images") #place to val

model_path = "last_epoch_weights.pth" #"pretrained-hardnet.pth"
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
video = False
half = False 
cpu = False 
trace = False 
openvino_exp = False 
export_onnx = False 

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
    input_height = 512
    input_width = 512
    dummy_input = torch.randn(1, 3, input_height, input_width).to(device)
    model =  ov.compile_model(ov.convert_model(model, example_input=dummy_input))


if export_onnx:
    input_width = 512
    input_height = 512
    torch_input = torch.randn(1, 3, input_width, input_height)
    onnx_program = torch.onnx.dynamo_export(model, torch_input)
    #onnx_program.save("mbv2_shufflenet_widerface.onnx")

save_xml = True
if save_xml :
    if not os.path.isdir(os.path.join(folder,"annos")):
        os.mkdir(os.path.join(folder,"annos"))
    from converter import Converter
    convert = Converter(os.path.join(folder,"annos"))
    

#video_path = r"E:\ESHOT 2024\Geshot Gediz 2 Garaji Gorselleri\Videolar\Otokar Ters Depo.avi"
video_path = r"G:\0010.mp4" #0 #r"G:\8_2023-07-31-11.37.01_novis_output.avi"
if video:
    cap = cv2.VideoCapture(video_path)
    while 1:
        ret,img = cap.read()
        #img = cv2.resize(img,(1280,720))
        img = img[...,::-1]
        image,annos = infer_image(model,img,classes,conf,half,input_shape=(320,320),cpu=cpu,openvino_exp=openvino_exp)
        image = cv2.resize(image,(1280,720))
        #print(annos)
        cv2.imshow("img",image[...,::-1])
        ch = cv2.waitKey(1)
        if ch == ord("q"): break


else:
    files = glob(folder+"/*.jpg") + glob(folder+"/*.png") + glob(folder+"/*.JPG")
    for i in files:
        print(i)
        if save_xml:
            annotations = []
        image,annos = infer_image(model,i,classes,conf,half,input_shape=(512,512),cpu=cpu,openvino_exp=openvino_exp)
        if save_xml:
            for b in annos:
                xmin = b[0]
                ymin = b[1]
                xmax = b[2]
                ymax = b[3]
                class_ = b[4]
                annotations.append([xmin,ymin,xmax,ymax,class_])
        image = cv2.resize(image,(1280,720))
        cv2.imshow("img",image[...,::-1])
        ch = cv2.waitKey(0)
        if ch == ord("q"): break
        if ch == ord("s"): 
            if save_xml:
                size_ = cv2.imread(i).shape
                convert(i.split("\\")[-1],size_,annotations)
            else:
                continue
