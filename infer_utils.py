import torch 
import numpy as np
from dataloader import preprocess_input,resize_image,cvtColor,resize_numpy
from utils_bbox import decode_bbox,postprocess
from PIL import Image 
import cv2 
import time 



def infer_image(model,img,classes,confidence=0.05,half=False,input_shape = (512,512)):
    #<class_name> <confidence> <left> <top> <right> <bottom>
    #files = glob(folder + "val_images/*.jpg") + glob(folder + "val_images/*.png")
    fps1 = time.time()
    if type(img) == str:
        image =  Image.open(img) 
    else:
        image = img #faster
        #image = Image.fromarray(img)
    
    image_shape = np.array(np.shape(image)[0:2])
    image  = cvtColor(image)
    #image_data = resize_image(image,tuple(input_shape),letterbox_image=True) 
    image_data = resize_numpy(image,tuple(input_shape),letterbox_image=True)


    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    image = np.array(image)
    fpre = time.time()
    print(f"Preprocessing took: {fpre - fps1} ms")

    try:
        f1 = time.time()
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor).cuda()
            if half: images = images.half()
            hm,wh,offset = model(images)
        if half: 
            hm  = hm.half()
            wh = wh.half()
            offset = offset.half()
        f2 = time.time()
        print(f"Model inference time: {f2-f1} ms , FPS: {1 / (f2-f1)}")

        fp1 = time.time()

        outputs = decode_bbox(hm,wh,offset,confidence=confidence)
        results = postprocess(outputs,True,image_shape,input_shape, True, 0.3) #letterbox true

        fp2 = time.time()
        print(f"Postprocessing took: {fp2-fp1} ms")
        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        for (conf,label,box) in zip(top_conf,top_label,top_boxes):
            ymin = box[0]
            xmin = box[1] 
            ymax = box[2] 
            xmax = box[3] 

            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            class_label = label
            name = classes[class_label]
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            cv2.putText(image,str(name),(xmin-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
            cv2.putText(image,str(conf),(xmax-3,ymin),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

            
    except Exception as e:
        print("Excepton:",e)
        pass
    fps2 = time.time()
    fps = 1 / (fps2-fps1) 
    cv2.putText(image,f'FPS:{fps:.2f}',(200,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
        #print(f"Could not infer an error occured: {e}")

    return image
            
def load_model(model,model_path):
    device = "cuda"
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    return model
