import torch 
import numpy as np
import json 
from dataloader import preprocess_input,resize_image,cvtColor
from utils_bbox import decode_bbox,postprocess
import os 
from PIL import Image 
import cv2 
from tqdm import tqdm 
def calculate_eval(model,cocoGt,classes,folder):
    #<class_name> <confidence> <left> <top> <right> <bottom>
    #files = glob(folder + "val_images/*.jpg") + glob(folder + "val_images/*.png")
    input_shape = (512,512)
    coco_format = []
    for i in tqdm(cocoGt.dataset["images"]):     
        id_ = i["id"]
        img_ = os.path.join(folder,"val_images",i["file_name"])
        image =  Image.open(img_) 
        image_shape = np.array(np.shape(image)[0:2])
        image  = cvtColor(image)
        image_data = resize_image(image,tuple(input_shape),letterbox_image=True)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image = np.array(image)
    
        try:
            with torch.no_grad():
                images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor).cuda()
                hm,wh,offset,iou_pred = model(images)

            outputs = decode_bbox(hm,wh,offset,iou_pred,confidence=0.05)
            results = postprocess(outputs,True,image_shape,input_shape, True, 0.3) #letterbox true
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
                #cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                class_label = label
                name = classes[class_label]
                anno = {"image_id": id_, "category_id": int(class_label)+1, "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], "score": float(conf)}
                coco_format.append(anno)

                
        except Exception as e:
            pass
            #print(f"Could not infer an error occured: {e}")
        #cv2.imwrite("img.jpg",image)
            
    with open('detection_results.json', 'w') as file:
        json.dump(coco_format, file)



def gt_check(img,hm,wh,offset):
    input_shape = (512,512)
    img_copy = np.transpose(np.array(img,dtype=np.float32()),(1,2,0))
    # image =  Image.open(img_) 
    image_shape = np.array(np.shape(img_copy)[0:2])
    print(image_shape)
    # image  = cvtColor(image)  
    # image_data = resize_image(image,input_shape,letterbox_image=True)
    # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    hm = torch.tensor(hm[None]).float().permute(0,3,1,2)
    wh = torch.tensor(wh[None]).float().permute(0,3,1,2)
    offset = torch.tensor(offset[None]).float().permute(0,3,1,2)

    outputs = decode_bbox(hm,wh,offset,confidence=0.05,cuda=False)
    results = postprocess(outputs,True,image_shape,input_shape, False, 0.3) 
    try:
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
            cv2.rectangle(img_copy,(xmin,ymin),(xmax,ymax),(0,255,0),2)
    except Exception as e:
        print("Exception is :",e)
    return img_copy
