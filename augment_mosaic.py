import random
import os
import xml.etree.ElementTree as ET
import cv2
import os
import glob
import numpy as np
from PIL import Image
import copy
from random import sample
from tqdm import tqdm 

OUTPUT_SIZE = (320, 320)  # Height, Width
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = 1 / 50  # if height or width lower than this scale, drop it.

Origin_VOCdevkit_path   = "VOCdevkit_Origin"
Out_VOCdevkit_path      = "VOCdevkit"
Origin_JPEGImages_path  = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages")
Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations")

ANNO_DIR = Origin_Annotations_path #'dataset/WiderPerson/Annotations/'
IMG_DIR = Origin_JPEGImages_path #'dataset/WiderPerson/Images/'
NUM_IMAGES = 250

#category_name = ['background', 'person']


def create_voc_xml(image_name, image_width, image_height, boxes, class_names):
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder").text = "images"
    filename = ET.SubElement(annotation, "filename").text = image_name
    
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
        xmin_abs = int(xmin * image_width)
        ymin_abs = int(ymin * image_height)
        xmax_abs = int(xmax * image_width)
        ymax_abs = int(ymax * image_height)
        
        ET.SubElement(bndbox, "xmin").text = str(xmin_abs)
        ET.SubElement(bndbox, "ymin").text = str(ymin_abs)
        ET.SubElement(bndbox, "xmax").text = str(xmax_abs)
        ET.SubElement(bndbox, "ymax").text = str(ymax_abs)
    
    return annotation

def save_voc_xml(annotation, output_path):
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def get_classes(sample_xmls, Origin_Annotations_path):
    unique_labels  = []
    for xml in sample_xmls:
        in_file = open(os.path.join(Origin_Annotations_path, xml), encoding='utf-8')
        tree    = ET.parse(in_file)
        root    = tree.getroot()
        
        for obj in root.iter('object'):
            cls     = obj.find('name').text
            if cls not in unique_labels:
                unique_labels.append(cls)
    return unique_labels

xml_names = os.listdir(Origin_Annotations_path)
sample_xmls     = sample(xml_names, len(xml_names))
category_name = get_classes(sample_xmls, Origin_Annotations_path)

def convert_annotation(xml_path, classes,img_w,img_h): #jpg_path,
    in_file = open(xml_path, encoding='utf-8')
    tree    = ET.parse(in_file)
    root    = tree.getroot()
    annos = []
    #line = copy.deepcopy(jpg_path)
    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None and hasattr(obj, "text"):
            difficult = obj.find('difficult').text
        if int(difficult)==1:
            continue
        
        cls     = obj.find('name').text
        cls_id = classes.index(cls)
        
        xmlbox  = obj.find('bndbox')
        xmin = int(float(xmlbox.find('xmin').text))
        ymin = int(float(xmlbox.find('ymin').text))
        xmax = int(float(xmlbox.find('xmax').text))
        ymax = int(float(xmlbox.find('ymax').text))
        
        xmin = max(xmin, 0) / img_w
        ymin = max(ymin, 0) / img_h
        xmax = min(xmax, img_w) / img_w
        ymax = min(ymax, img_h) / img_h
        b = [cls_id,xmin,ymin,xmax,ymax]
        annos.append(b)
        #line += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
    return annos


def main():
    img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)
    print("Augmenting")
    for i in tqdm(range(NUM_IMAGES)):
        idxs = random.sample(range(len(annos)), 4)


        new_image, new_annos = update_image_and_anno(img_paths, annos,
                                                    idxs,
                                                    OUTPUT_SIZE, SCALE_RANGE,
                                                    filter_scale=FILTER_TINY_SCALE)

        cv2.imwrite(f'img/jpg/{i}_mosaic_output.jpg', new_image)
        for anno in new_annos:
            start_point = (int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0]))
            end_point = (int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0]))
            cv2.rectangle(new_image, start_point, end_point, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(f'img/{i}_mosaic_output_box.jpg', new_image)

        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        new_image = Image.fromarray(new_image.astype(np.uint8))
        xml_annotation = create_voc_xml(f'{i}_mosaic_output.jpg', OUTPUT_SIZE[0], OUTPUT_SIZE[1], new_annos, category_name)
        save_voc_xml(xml_annotation, f'img/anno/{i}_mosaic_output.xml')

    #new_image.show() #opens gimp


def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        img_annos = all_annos[idx]
        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = bbox[2] * scale_y
                xmax = bbox[3] * scale_x
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = bbox[2] * scale_y
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = bbox[4] * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] * scale_x
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = bbox[3] * scale_x
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])
        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = scale_x + bbox[1] * (1 - scale_x)
                ymin = scale_y + bbox[2] * (1 - scale_y)
                xmax = scale_x + bbox[3] * (1 - scale_x)
                ymax = scale_y + bbox[4] * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    if 0 < filter_scale:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    return output_img, new_anno


def get_dataset(anno_dir, img_dir):
    #class_id = category_name.index('person')

    img_paths = []
    annos = []
    for anno_file in tqdm(glob.glob(os.path.join(anno_dir, '*.xml'))):
        anno_id = anno_file.split('/')[-1].split('.')[0]

        #with open(anno_file, 'r') as f:
        #    num_of_objs = int(f.readline())
        #print(os.path.join(img_dir, f'{anno_id}.'))
        
        img_path = os.path.join(img_dir, f'{anno_id}.png')
        img = cv2.imread(img_path)
        if img is None:
            img_path = os.path.join(img_dir, f'{anno_id}.jpg')
            img = cv2.imread(img_path)
            
        img_height, img_width, _ = img.shape
        del img
        #anno_id = os.path.join(anno_dir,anno_id,".xml")
        boxes = convert_annotation(anno_file,category_name,img_width,img_height)
        # boxes = []
        # for _ in range(num_of_objs):
        #     obj = f.readline().rstrip().split(' ')
        #     obj = [int(elm) for elm in obj]
        #     if 3 < obj[0]:
        #         continue

        #     xmin = max(obj[1], 0) / img_width
        #     ymin = max(obj[2], 0) / img_height
        #     xmax = min(obj[3], img_width) / img_width
        #     ymax = min(obj[4], img_height) / img_height

        #     boxes.append([class_id, xmin, ymin, xmax, ymax])

        # if not boxes:
        #     continue

        img_paths.append(img_path)
        annos.append(boxes)
    return img_paths, annos


if __name__ == '__main__':
    main()