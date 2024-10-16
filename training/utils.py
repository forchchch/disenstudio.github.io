import os
import logging
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import random

scene_prompt = ["on the beach", "in the ocean", "in the flowers", "on the grass", "in the desert", "under Eiffel Tower", "on the Great Wall", "at the Times Square"]
action_prompt = ["run", "walk", "jump", "cook", "play football", "play basketball", "play the guitar"]

def make_prior_prompt(original_prompt):
    random_scene = random.choice(scene_prompt)
    context_prompt, region_prompt_list = original_prompt
    new_region_prompt_list = []
    for prompt_region in region_prompt_list:
        r_prompt, r_region = prompt_region
        random_action = random.choice(action_prompt)
        new_r_prompt = r_prompt + " " + random_action + "; best quality"
        new_region_prompt_list.append((new_r_prompt, r_region))
    return (random_scene, new_region_prompt_list)
    
def my_make_dir(out_root, exp_name):
    logging_dir = os.path.join(out_root, exp_name, "loggers")
    out_image_dir =  os.path.join(out_root, exp_name, "generated_images")
    out_check_dir = os.path.join(out_root, exp_name, "checkpoint")
        
    os.makedirs(logging_dir, exist_ok=True)    
    os.makedirs(out_image_dir, exist_ok=True) 
    os.makedirs(out_check_dir, exist_ok=True) 
    
    return logging_dir, out_image_dir, out_check_dir

def get_logger(filename,name):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_torch_model(model, path):
    torch.save(model.state_dict() ,path)

def gen_mix_image(origin_path, name_list, save_path, target_size=512, use_copy=False):
    folder_name = ""
    for name in name_list:
        if name.startswith("instance/"):
            _,name = name.split("/")
        if len(folder_name)==0:
            folder_name += name
        else:
            folder_name += "_"
            folder_name += name
    save_dir = os.path.join(save_path, folder_name, "size_"+str(target_size))
    if not os.path.exists(save_dir) or len(os.listdir(save_dir))==0:
        os.makedirs(save_dir, exist_ok=True)
        path_list = [os.path.join(origin_path, name) for name in name_list]
        subject_num = len(name_list)
        image_num_list = []
        for path in path_list:
            image_num = len(os.listdir(path))
            image_num_list.append(image_num)
        max_image_num = max(image_num_list)
        
        for i in range(max_image_num):
            target = Image.new('RGB', (512*subject_num,512))
            for j,path in enumerate(path_list):
                image_list = os.listdir(path)
                image_list.sort()
                image_path = os.path.join(path, image_list[i%len(image_list)])
                image = Image.open(image_path)
                if not image.mode == "RGB":
                    image = image.convert("RGB") 
                image = image.resize((512,512))
                target.paste(image, (0 + j*512,0))                
            target = target.resize((target_size,512))        
            target.save(os.path.join(save_dir, str(i)+".jpg"))
            if use_copy:
                target.save(os.path.join(save_dir, str(i+max_image_num)+".jpg"))
    return save_dir

def mix_background(mix_image_path, mix_mask_path, all_background_path, save_path, target_size=512, use_copy=False):
    name_list = mix_image_path.split("/")
    folder_name = name_list[-2]
    save_dir = os.path.join(save_path, folder_name, "size_"+str(target_size))
    os.makedirs(save_dir, exist_ok=True) 
    if not os.path.exists(save_dir) or len(os.listdir(save_dir))==0:
        image_list = os.listdir(mix_image_path)
        image_list.sort()
        mask_list = os.listdir(mix_mask_path)
        mask_list.sort()
        background_list = os.listdir(all_background_path)
        
        image_num = len(image_list)
        for i in range(image_num):
            image_path = os.path.join(mix_image_path, image_list[i])
            image = cv2.imread(image_path)
            
            mask_path = os.path.join(mix_mask_path, mask_list[i])
            mask = cv2.imread(mask_path)   
            
            background_path =  os.path.join(all_background_path, background_list[i%len(background_list)])
            background = cv2.imread(background_path)
            background = cv2.resize(background, dsize=(image.shape[1], image.shape[0]) )
            
            mask1 = np.sum(mask, axis=2, keepdims=True)
            mask2 = (mask1 > 100.0) + 0.0
            
            new_image = mask2*image + (1.0-mask2)*background
            cv2.imwrite( os.path.join(save_dir, str(i)+".jpg"), new_image )
            if use_copy:
                cv2.imwrite( os.path.join(save_dir, str(i+image_num)+".jpg"), image )
        
    return save_dir