import os
import random

#input folder
input_folder = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/folds/wsol-done-right-splits/CAMELYON512/fold-0/train/'
class_labels_file = os.path.join(input_folder, 'class_labels.txt')
image_ids_file = os.path.join(input_folder, 'image_ids.txt')
images_size_file = os.path.join(input_folder, 'images_size.txt')
localization_file = os.path.join(input_folder, 'localization.txt')

#create output folder
output_folder = '/export/livia/home/vision/Aguichemerre/Pixel-Adaptation/folds/wsol-done-right-splits/CAMELYON512/fold-6/train/'
os.makedirs(output_folder, exist_ok=True)


#number of samples selected per class
m = 250

#select image ids from image_ids.txt
with open(image_ids_file, 'r') as f:
    image_ids = f.readlines()

#select image ids from class_labels.txt (split per class)
cancer_paths = image_ids[:len(image_ids)//2]
normal_paths = image_ids[len(image_ids)//2:]

#select m random samples from each class
selected_cancer_paths = random.sample(cancer_paths, m)
selected_normal_paths = random.sample(normal_paths, m)


selected_paths = selected_cancer_paths + selected_normal_paths


with open(os.path.join(output_folder, 'image_ids.txt'), 'w') as f:
    f.writelines(selected_paths)

with open(os.path.join(output_folder, 'class_labels.txt'), 'w') as f:
    for path in selected_cancer_paths:
        f.write(f"{path.strip()},1\n")
    for path in selected_normal_paths:
        f.write(f"{path.strip()},0\n")

with open(os.path.join(output_folder, 'images_size.txt'), 'w') as f:
    for path in selected_paths:
        f.write(f"{path.strip()},512,512\n")


localization_dict = {}
with open(localization_file, 'r') as f:
    for line in f:
        key = line.split(',')[0]
        localization_dict[key] = line

with open(os.path.join(output_folder, 'localization.txt'), 'w') as f:
    for path in selected_paths:
        key = path.strip()
        if key in localization_dict:
            f.write(localization_dict[key])

print(f'Files created in {output_folder}')

