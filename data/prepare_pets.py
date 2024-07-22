import os
import random
import json
import shutil

root = "data/oxford-iiit-pet/images"
x = os.listdir(root)
class_label_map = {}

j=0
for i in range(len(x)):
    if "_".join(x[i].split(".")[0].split("_")[:-1]) not in class_label_map:
        class_label_map["_".join(x[i].split(".")[0].split("_")[:-1])] = j
        j+=1

class_imgs = {}
for i in range(len(x)):
    class_name = "_".join(x[i].split(".")[0].split("_")[:-1]) 
    if class_name not in class_imgs:
        class_imgs[class_name] = [f"{root}/{x[i]}"]
    else:
        class_imgs[class_name].append(f"{root}/{x[i]}")

for key in class_imgs.keys():
    print(key,":",len(class_imgs[key]))



images_dict = class_imgs
images_folder = "/".join(root.split("/")[:-1])

for label, image_paths in images_dict.items():
    label_dir = os.path.join(images_folder, label)
    os.makedirs(label_dir, exist_ok=True)
    
    for image_path in image_paths:
        # Get the full path of the image
        full_image_path = os.path.join(image_path)
        
        # Move the image to the label's subdirectory
        if os.path.exists(full_image_path):
            shutil.move(full_image_path, label_dir)
        else:
            print(f"Image not found: {full_image_path}")

print("Images have been moved to their respective subdirectories.")