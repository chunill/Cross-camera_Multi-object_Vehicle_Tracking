# Cross-camera_Multi-object_Vehicle_Tracking 

此專案使用該競賽的資料集：https://tbrain.trendmicro.com.tw/Competitions/Details/32

## Installation

**The code was tested on Ubuntu 20.04**

BoT-SORT code is based on ByteTrack and FastReID. <br>
Visit their installation guides for more setup options.
 
### Setup with Conda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n botsort python=3.8
conda activate botsort
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch 2.2.1 and torchvision==0.17.1 

**Step 3.**
1. Fork this Repository and clone your Repository to your device.

2. 將`fast_reid\datasets\AICUP-ReID`裡的三個.gitkeep刪除以免之後操作發生意外錯誤

![gitkeep](https://i.imgur.com/iTKZN1e.png)

**Step 4.** **Install numpy first!!**
```shell
pip install numpy
```

**Step 5.** Install `requirements.txt`
```shell
pip install -r requirements.txt
```

**Step 6.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 7.** Others
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```

## Data Preparation

Download the AI_CUP dataset, the original dataset structure is:
```python
├── train
│   ├── images
│   │   ├── 0902_150000_151900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── 0902_190000_191900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── ...
│   └── labels
│   │   ├── 0902_150000_151900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.txt (CamID_FrameNum)
│   │   │  ├── 0_00002.txt
│   │   │  ├── ...
│   │   │  ├── 1_00001.txt (CamID_FrameNum)
│   │   │  ├── 1_00002.txt
│   │   │  ├── ...
│   │   │  ├── 7_00001.txt (CamID_FrameNum)
│   │   │  ├── 7_00002.txt
│   │   ├── 0902_190000_191900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.txt (CamID_FrameNum)
│   │   │  ├── 0_00002.txt
│   │   │  ├── ...
│   │   │  ├── 1_00001.txt (CamID_FrameNum)
│   │   │  ├── 1_00002.txt
│   │   │  ├── ...
│   │   │  ├── 7_00001.txt (CamID_FrameNum)
│   │   │  ├── 7_00002.txt
│   │   ├── ...
--------------------------------------------------
├── test
│   ├── images
│   │   ├── 0902_150000_151900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── 0902_190000_191900 (Timestamp: Date_StartTime_EndTime)
│   │   │  ├── 0_00001.jpg (CamID_FrameNum)
│   │   │  ├── 0_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 1_00001.jpg (CamID_FrameNum)
│   │   │  ├── 1_00002.jpg
│   │   │  ├── ...
│   │   │  ├── 7_00001.jpg (CamID_FrameNum)
│   │   │  ├── 7_00002.jpg
│   │   ├── ...
```

and download [VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild) datasets(只有訓練ReId時會用到該datasets，需要自己去申請datasets).

需下載該images資料夾裡的所有檔案，該頁面的url格式為 https://drive.google.com/drive/folders/id ，複製id的部分並貼到腳本的`id`參數，最後輸入 `output_pth` 就可以執行腳本。
![VERI-Wild download](https://i.imgur.com/yH077sQ.png)

```python
# download_VERI-Wild.py

import gdown

id = "input your download id"
output_pth = "input your output path"

gdown.download_folder(id=id, output=output_pth, quiet=False)
```

執行畫面：
![download demo](https://i.imgur.com/KZ415qD.png)

載完記得解壓縮。

### Prepare ReID Dataset

For training the ReID, detection patches must be generated as follows:   

```shell
cd <BoT-SORT_dir>

# For AICUP datasets
python fast_reid/datasets/generate_AICUP_patches.py --data_path <dataets_dir>/AI_CUP_MCMOT_dataset/train
```
將AICUP的資料處理好後要來處理VeRI_Wild的資料集
```python
python3 convert_format.py -s VeRI_Wild資料集解壓縮完的路徑
```
該指令會把VeRI_Wild資料集的圖片都改成特定格式的名稱並輸出到`fast_reid/datasets/AICUP-ReID/all_datasets`


最後執行`fast_reid\datasets\set_train_data.py`，將AICUP和VeRI_Wild的圖片依照list的內容複製到bounding_box_train和bounding_box_test資料夾。
```python
# fast_reid\datasets\set_train_data.py

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
```

> [!TIP]
> You can link dataset to FastReID ```export FASTREID_DATASETS=<BoT-SORT_dir>/fast_reid/datasets```. If left unset, the default is `fast_reid/datasets` 


### Prepare YOLOv7 Dataset

> [!WARNING]
> We only implemented the fine-tuning interface for `yolov7`
> If you need to change the object detection model, please do it yourself.

run the `yolov7/tools/AICUP_to_YOLOv7.py` by the following command:
```
cd <BoT-SORT_dir>
python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir datasets/AI_CUP_MCMOT_dataset/train --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo
```
The file tree after conversion by `AICUP_to_YOLOv7.py` is as follows:

```python
/datasets/AI_CUP_MCMOT_dataset/yolo
    ├── train
    │   ├── images
    │   │   ├── 0902_150000_151900_0_00001.jpg (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 0902_150000_151900_0_00002.jpg
    │   │   ├── ...
    │   │   ├── 0902_150000_151900_7_00001.jpg
    │   │   ├── 0902_150000_151900_7_00002.jpg
    │   │   ├── ...
    │   └── labels
    │   │   ├── 0902_150000_151900_0_00001.txt (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 0902_150000_151900_0_00002.txt
    │   │   ├── ...
    │   │   ├── 0902_150000_151900_7_00001.txt
    │   │   ├── 0902_150000_151900_7_00002.txt
    │   │   ├── ...
    ├── valid
    │   ├── images
    │   │   ├── 1015_190000_191900_0_00001.jpg (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 1015_190000_191900_0_00002.jpg
    │   │   ├── ...
    │   │   ├── 1015_190000_191900_7_00001.jpg
    │   │   ├── 1015_190000_191900_7_00002.jpg
    │   │   ├── ...
    │   └── labels
    │   │   ├── 1015_190000_191900_0_00001.txt (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 1015_190000_191900_0_00002.txt
    │   │   ├── ...
    │   │   ├── 1015_190000_191900_7_00001.txt
    │   │   ├── 1015_190000_191900_7_00002.txt
    │   │   ├── ...
```

## Training (Fine-tuning)

### Train the ReID Module for AICUP

After generating the AICUP ReID dataset as described in the 'Data Preparation' section.

```shell
cd <BoT-SORT_dir>

# For training AICUP 
python3 fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
```

The training results are stored by default in ```logs/AICUP/bagtricks_R50-ibn``` (最好的model是model_0005.pth).

Refer to [FastReID](https://github.com/JDAI-CV/fast-reid) repository for additional explanations and options.

> [!IMPORTANT]  
> Since we did not generate the `query` and `gallery` datasets required for evaluation when producing the ReID dataset (`MOT17_ReID` provided by BoT-SORT also not provide them), please skip the following TrackBack when encountered after training completion.

```shell
Traceback (most recent call last):
...
File "./fast_reid/fastreid/evaluation/reid_evaluation.py", line 107, in evaluate
    cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
  File "./fast_reid/fastreid/evaluation/rank.py", line 198, in evaluate_rank
    return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
  File "rank_cy.pyx", line 20, in rank_cy.evaluate_cy
  File "rank_cy.pyx", line 28, in rank_cy.evaluate_cy
  File "rank_cy.pyx", line 240, in rank_cy.eval_market1501_cy
AssertionError: Error: all query identities do not appear in gallery
```

### Fine-tune YOLOv7 for AICUP
Single GPU finetuning for AICUP dataset
[`yolov7-e6e_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt)

``` shell
cd <BoT-SORT_dir>
# finetune p5 models
python yolov7/train.py --device 0 --batch-size 8 --epochs 100 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-AICUP.yaml --weights 'pretrained/yolov7-e6e_training.pt' --name yolov7-AICUP --hyp data/hyp.scratch.custom.yaml
```
The training results will be saved by default at `runs/train`(最好的model是best.pt).

## Citation

```
@article{aharon2022bot,
  title={BoT-SORT: Robust Associations Multi-Pedestrian Tracking},
  author={Aharon, Nir and Orfaig, Roy and Bobrovsky, Ben-Zion},
  journal={arXiv preprint arXiv:2206.14651},
  year={2022}
}
```


## Acknowledgement

A large part of the codes, ideas and results are borrowed from
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)
- [FastReID](https://github.com/JDAI-CV/fast-reid)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLOv7](https://github.com/wongkinyiu/yolov7)

Thanks for their excellent work!











