import os
from tqdm import tqdm
from PIL import Image 
import argparse

def convert_images_to_bmp(source_path, target_path):
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                img = Image.open(full_path)
                folder_name = os.path.basename(root)
                base_name = os.path.splitext(file)[0]
                new_name = f'{folder_name}_{base_name}.bmp'
                img.save(os.path.join(target_path, new_name), 'BMP')
                print(new_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert the file format of the dataset image and rename it.')
    parser.add_argument('-s', '--source', help='Source folder path', required=True)
    parser.add_argument('-t', '--target', help='Target folder path', default="fast_reid/datasets/AICUP-ReID/all_datasets")
    args = parser.parse_args()
    
    convert_images_to_bmp(args.source, args.target)
