from PIL import Image, ImageDraw #version 6.1.0
import PIL #version 1.2.0
import torch
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2
from glob import glob
from tqdm import tqdm 

voc_labels = [i.replace("\n","") for i in open("classes.txt","r")]
label_map = {k: v+1 for v, k in enumerate(voc_labels)}
#Inverse mapping
rev_label_map = {v: k for k, v in label_map.items()}
#Colormap for bounding box
CLASSES = 20
distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(CLASSES)]
label_color_map  = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def parse_annot(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    
    for object in root.iter("object"):
        difficult = int(object.find("difficult").text == "1")
        label = object.find("name").text.upper().strip()
        if label not in label_map:
            print("{0} not in label map.".format(label))
            assert label in label_map
            
        bbox =  object.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
        
    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}

def draw_PIL_image(image, boxes, labels):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    labels = labels.tolist()
    draw = ImageDraw.Draw(new_image)
    boxes = boxes.tolist()
    for i in range(len(boxes)):
        draw.rectangle(xy= boxes[i], outline= label_color_map[rev_label_map[labels[i]]])
        draw.text(xy=(boxes[i][0],boxes[i][1]),text = str(rev_label_map[labels[i]]))

    
    #display(new_image)
    new_image = np.array(new_image)[...,::-1]
    cv2.imshow("new_image",new_image)
    ch = cv2.waitKey(0)
    if ch == "q": cv2.destroyWindow("new_image")


def mixup(image_info_1, image_info_2, lambd):
    '''
        Mixup 2 image
        
        image_info_1, image_info_2: Info dict 2 image with keys = {"image", "label", "box", "difficult"}
        lambd: Mixup ratio
        
        Out: mix_image (Temsor), mix_boxes, mix_labels, mix_difficulties
    '''
    img1 = image_info_1["image"]    #Tensor
    img2 = image_info_2["image"]    #Tensor
    mixup_width = max(img1.shape[2], img2.shape[2])
    mix_up_height = max(img1.shape[1], img2.shape[1])
    
    mix_img = torch.zeros(3, mix_up_height, mixup_width)
    mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * lambd
    mix_img[:, :img2.shape[1], :img2.shape[2]] += img2 * (1. - lambd)
    
    mix_labels = torch.cat((image_info_1["label"], image_info_2["label"]), dim= 0)
    
    mix_difficulties = torch.cat((image_info_1["difficult"], image_info_2["difficult"]), dim= 0)
    
    mix_boxes = torch.cat((image_info_1["box"], image_info_2["box"]), dim= 0)
    
    return mix_img, mix_boxes, mix_labels, mix_difficulties


def create_voc_xml(image_name, image_width, image_height, boxes, class_names):
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder").text = image_name.split("/")[0] #"images"
    filename = ET.SubElement(annotation, "filename").text = image_name.split("/")[-1]
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_width)
    ET.SubElement(size, "height").text = str(image_height)
    ET.SubElement(size, "depth").text = "3"
    
    for box in boxes:
        obj = ET.SubElement(annotation, "object")
        cls_id, xmin, ymin, xmax, ymax = box
        cls_name = class_names[int(cls_id)]
        
        ET.SubElement(obj, "name").text = cls_name
        bndbox = ET.SubElement(obj, "bndbox")
        
        # Convert YOLO format (normalized) to VOC format (absolute)
        xmin_abs = int(xmin) 
        ymin_abs = int(ymin)
        xmax_abs = int(xmax)
        ymax_abs = int(ymax)
        
        ET.SubElement(bndbox, "xmin").text = str(xmin_abs)
        ET.SubElement(bndbox, "ymin").text = str(ymin_abs)
        ET.SubElement(bndbox, "xmax").text = str(xmax_abs)
        ET.SubElement(bndbox, "ymax").text = str(ymax_abs)
    
    return annotation

def save_voc_xml(annotation, output_path):
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

# image1 = Image.open(r"C:\Users\John\Desktop\derpetv5_xml\val_images\0_vlcsnap-2023-04-06-15h26m53s907.png", mode= "r")
# image1 = image1.convert("RGB")
# objects1= parse_annot(r"C:\Users\John\Desktop\derpetv5_xml\val_images\0_vlcsnap-2023-04-06-15h26m53s907.xml")
# boxes1 = torch.FloatTensor(objects1['boxes'])
# labels1 = torch.LongTensor(objects1['labels']) 
# difficulties1 = torch.ByteTensor(objects1['difficulties'])
# draw_PIL_image(image1, boxes1, labels1)



# image2 = Image.open(r"C:\Users\John\Desktop\derpetv5_xml\val_images\1_vlcsnap-2023-04-06-11h27m09s102.png", mode= "r")
# image2 = image2.convert("RGB")
# objects2= parse_annot(r"C:\Users\John\Desktop\derpetv5_xml\val_images\1_vlcsnap-2023-04-06-11h27m09s102.xml")
# boxes2 = torch.FloatTensor(objects2['boxes'])
# labels2 = torch.LongTensor(objects2['labels']) 
# difficulties2 = torch.ByteTensor(objects2['difficulties'])
# draw_PIL_image(image2, boxes2, labels2)


# image_info_1 = {"image": F.to_tensor(image1), "label": labels1, "box": boxes1, "difficult": difficulties1}
# image_info_2 = {"image": F.to_tensor(image2), "label": labels2, "box": boxes2, "difficult": difficulties2}

# lambd = random.uniform(0, 1)
# mix_img, mix_boxes, mix_labels, mix_difficulties = mixup(image_info_1, image_info_2, lambd)
# draw_PIL_image(F.to_pil_image(mix_img), mix_boxes, mix_labels)
# print("Lambda: ",lambd)

path = r"C:\Users\John\Desktop\derpetv5_xml\val_images"
folder = glob(os.path.join(path,"*.png")) + glob(os.path.join(path,"*.jpg"))
num_outputs = 200
#OUTPUT_SIZE = (512,512)

for i in tqdm(range(num_outputs)):
    rand_1 = random.randint(0,len(folder) - 1)
    rand_2 = random.randint(0,len(folder) - 1)
    if rand_1 == rand_2: continue 
    image1 = Image.open(folder[rand_1], mode= "r")
    image1 = image1.convert("RGB")
    objects1= parse_annot(folder[rand_1].replace(".jpg",".xml").replace(".png",".xml"))
    boxes1 = torch.FloatTensor(objects1['boxes'])
    labels1 = torch.LongTensor(objects1['labels']) 
    difficulties1 = torch.ByteTensor(objects1['difficulties'])

    image2 = Image.open(folder[rand_2], mode= "r")
    image2 = image2.convert("RGB")
    objects2= parse_annot(folder[rand_2].replace(".jpg",".xml").replace(".png",".xml"))
    boxes2 = torch.FloatTensor(objects2['boxes'])
    labels2 = torch.LongTensor(objects2['labels']) 
    difficulties2 = torch.ByteTensor(objects2['difficulties'])

    image_info_1 = {"image": F.to_tensor(image1), "label": labels1, "box": boxes1, "difficult": difficulties1}
    image_info_2 = {"image": F.to_tensor(image2), "label": labels2, "box": boxes2, "difficult": difficulties2}

    lambd = random.uniform(0, 1)
    mix_img, mix_boxes, mix_labels, mix_difficulties = mixup(image_info_1, image_info_2, lambd)
    mix_img = F.to_pil_image(mix_img)
    new_anno = []
    for b,l in zip(mix_boxes,mix_labels):
        b = b.numpy().astype("int")
        l = int(l.numpy())
        new_anno.append([l,b[0],b[1],b[2],b[3]])
    #print(new_anno)
    xml_annotation = create_voc_xml(f'mixups/{i}_mixup_output.jpg', np.array(mix_img).shape[1],  np.array(mix_img).shape[0] , new_anno, rev_label_map)
    save_voc_xml(xml_annotation, f'mixups/{i}_mixup_output.xml')
    cv2.imwrite(f'mixups/{i}_mixup_output.jpg', np.array(mix_img)[...,::-1])
    #draw_PIL_image(mix_img, mix_boxes, mix_labels)
    #print("Lambda: ",lambd)



