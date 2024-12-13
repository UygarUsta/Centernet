import xml.etree.ElementTree as ET
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime 

def parse_xml(xml_path):
    """Parses Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
    return objects, width, height

def create_xml(image_name, image_width, image_height, objects, output_path):
    """Creates Pascal VOC XML annotation file."""
    root = ET.Element('annotation')

    folder = ET.SubElement(root, 'folder')
    folder.text = 'images'  # Or your folder name

    filename = ET.SubElement(root, 'filename')
    filename.text = image_name

    path = ET.SubElement(root, 'path')
    path.text = image_name  # Or the full path

    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_width)
    height = ET.SubElement(size, 'height')
    height.text = str(image_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    for obj in objects:
        object_elem = ET.SubElement(root, 'object')
        name = ET.SubElement(object_elem, 'name')
        name.text = obj['name']
        pose = ET.SubElement(object_elem, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object_elem, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(object_elem, 'difficult')
        difficult.text = '0'
        bndbox = ET.SubElement(object_elem, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(obj['bbox'][0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(obj['bbox'][1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(obj['bbox'][2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(obj['bbox'][3])

    tree = ET.ElementTree(root)
    tree.write(output_path)

def mixup_object_detection(image_path1, xml_path1, image_path2, xml_path2, output_image_path, output_xml_path, alpha=1.0):
    """Applies mixup augmentation for object detection with corrected bounding boxes."""

    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    objects1, w1, h1 = parse_xml(xml_path1)
    objects2, w2, h2 = parse_xml(xml_path2)

    # Resize images to have the same dimensions (using the larger dimensions)
    max_w = max(w1, w2)
    max_h = max(h1, h2)

    image1_resized = cv2.resize(image1, (max_w, max_h))
    image2_resized = cv2.resize(image2, (max_w, max_h))

    # Calculate resizing ratios
    w_ratio1 = max_w / w1
    h_ratio1 = max_h / h1
    w_ratio2 = max_w / w2
    h_ratio2 = max_h / h2
    
    # Ensure images are float for proper blending
    image1_resized = image1_resized.astype(np.float32)
    image2_resized = image2_resized.astype(np.float32)

    # Sample lambda from Beta distribution
    lmbda = np.random.beta(alpha, alpha)

    mixed_image = (1 - lmbda) * image1_resized + lmbda * image2_resized
    mixed_image = mixed_image.astype(np.uint8)

    mixed_objects = []

    # Correct bounding box coordinates for image 1
    for obj in objects1:
        xmin = int(obj['bbox'][0] * w_ratio1)
        ymin = int(obj['bbox'][1] * h_ratio1)
        xmax = int(obj['bbox'][2] * w_ratio1)
        ymax = int(obj['bbox'][3] * h_ratio1)
        mixed_objects.append({'name': obj['name'], 'bbox': [xmin, ymin, xmax, ymax]})

    # Correct bounding box coordinates for image 2
    for obj in objects2:
        xmin = int(obj['bbox'][0] * w_ratio2)
        ymin = int(obj['bbox'][1] * h_ratio2)
        xmax = int(obj['bbox'][2] * w_ratio2)
        ymax = int(obj['bbox'][3] * h_ratio2)
        mixed_objects.append({'name': obj['name'], 'bbox': [xmin, ymin, xmax, ymax]})

    cv2.imwrite(output_image_path, mixed_image)
    create_xml(os.path.basename(output_image_path), max_w, max_h, mixed_objects, output_xml_path)

# Example usage:
image_dir = "/mnt/e/derpetv5_xml/train_images" #replace with your image directory
annotation_dir = "/mnt/e/derpetv5_xml/train_images" #replace with your annotation directory
output_image_dir = "./output/images" #replace with your output image directory
output_annotation_dir = "./output/annotations" #replace with your output annotation directory

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.JPG'))]
xml_files = [f for f in os.listdir(annotation_dir) if f.lower().endswith('.xml')]

output_images = 350
current_dateTime = datetime.now()
out_name =  str(current_dateTime.hour) + "." + str(current_dateTime.minute) + "." + str(current_dateTime.day) + "." + str(current_dateTime.month) + "." + str(current_dateTime.year)

for i in tqdm(range(output_images)): # Example: perform 5 mixup augmentations
    img1_name = random.choice(image_files)
    img2_name = random.choice(image_files)
    
    xml1_name = img1_name.split(".")[0]+".xml"
    xml2_name = img2_name.split(".")[0]+".xml"

    if xml1_name not in xml_files or xml2_name not in xml_files:
        continue

    image_path1 = os.path.join(image_dir, img1_name)
    xml_path1 = os.path.join(annotation_dir, xml1_name)
    image_path2 = os.path.join(image_dir, img2_name)
    xml_path2 = os.path.join(annotation_dir, xml2_name)

    output_image_name = f"{out_name}_mixup_{i}.jpg"
    output_xml_name = f"{out_name}_mixup_{i}.xml"

    output_image_path = os.path.join(output_image_dir, output_image_name)
    output_xml_path = os.path.join(output_annotation_dir, output_xml_name)
    
    mixup_object_detection(image_path1, xml_path1, image_path2, xml_path2, output_image_path, output_xml_path)

print("Mixup augmentation complete.")