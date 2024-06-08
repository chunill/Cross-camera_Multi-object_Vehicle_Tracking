import shutil
from tqdm import tqdm

def read_file_list(file_path):
    with open(file_path, "r") as f:
        file_list = f.readlines()
    return [file_name.strip() for file_name in file_list]

def copy_file(list_path: str, target_file: str):
    files = read_file_list(list_path)

    for file in tqdm(files, desc="Processing images"):
        shutil.copy(f"fast_reid/datasets/AICUP-ReID/all_datasets/{file}", target_file)

copy_file("fast_reid/datasets/train_list.txt", "fast_reid/datasets/AICUP-ReID/bounding_box_train")
copy_file("fast_reid/datasets/test_list.txt", "fast_reid/datasets/AICUP-ReID/bounding_box_test")