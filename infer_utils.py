import torch 
import numpy as np
from dataloader import preprocess_input,resize_image,cvtColor,resize_numpy
from utils_bbox import decode_bbox,postprocess
from PIL import Image 
import cv2 
import time 



def infer_image(model,img,classes,confidence=0.05,half=False,input_shape = (320,320),cpu = False):
    #<class_name> <confidence> <left> <top> <right> <bottom>
    #files = glob(folder + "val_images/*.jpg") + glob(folder + "val_images/*.png")
    if cpu:
       device = torch.device("cpu")
       cuda = False 
    else:
       device = torch.device("cuda")
       cuda = True
       
    fps1 = time.time()
    if type(img) == str:
        image =  Image.open(img) 
    else:
        image = img #faster
        #image = Image.fromarray(img)
    
    image_shape = np.array(np.shape(image)[0:2])
    image  = cvtColor(image)
    #image_data = resize_image(image,tuple(input_shape),letterbox_image=True) 
    image_data = resize_numpy(image,tuple(input_shape),letterbox_image=False)


    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    image = np.array(image)
    fpre = time.time()
    print(f"Preprocessing took: {fpre - fps1} ms")
    box_annos = []
    try:
        f1 = time.time()
        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor).to(device)
            if half: images = images.half()
            hm,wh,offset = model(images)
        if half: 
            hm  = hm.half()
            wh = wh.half()
            offset = offset.half()
        f2 = time.time()
        print(f"Model inference time: {f2-f1} ms , FPS: {1 / (f2-f1)}")

        fp1 = time.time()

        outputs = decode_bbox(hm,wh,offset,confidence=confidence,cuda=cuda)
        results = postprocess(outputs,True,image_shape,input_shape, False, 0.3) #letterbox true

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
            box_annos.append([xmin,ymin,xmax,ymax,str(name),conf])
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

    return image,box_annos
            
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


def hardnet_load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model