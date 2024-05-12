# EfficientPS For Semantic Segmentation
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientps-efficient-panoptic-segmentation/panoptic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-cityscapes-val?p=efficientps-efficient-panoptic-segmentation) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientps-efficient-panoptic-segmentation/panoptic-segmentation-on-cityscapes-test)](https://paperswithcode.com/sota/panoptic-segmentation-on-cityscapes-test?p=efficientps-efficient-panoptic-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientps-efficient-panoptic-segmentation/panoptic-segmentation-on-mapillary-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-mapillary-val?p=efficientps-efficient-panoptic-segmentation)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientps-efficient-panoptic-segmentation/panoptic-segmentation-on-kitti-panoptic)](https://paperswithcode.com/sota/panoptic-segmentation-on-kitti-panoptic-segmentationl?p=efficientps-efficient-panoptic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficientps-efficient-panoptic-segmentation/panoptic-segmentation-on-indian-driving)](https://paperswithcode.com/sota/panoptic-segmentation-on-panoptic-segmentation-on-indian-driving?p=efficientps-efficient-panoptic)


# Introduction

EfficientPS is a state-of-the-art top-down approach for panoptic segmentation, where the goal is to assign semantic labels (e.g., car, road, tree and so on) to every pixel in the input image as well as instance labels (e.g. an id of 1, 2, 3, etc) to pixels belonging to thing classes.

This repository contains the **PyTorch implementation** of our IJCV'2021 paper [EfficientPS: Efficient Panoptic Segmentation](https://arxiv.org/abs/2004.02307). The repository builds on [mmdetection](https://github.com/open-mmlab/mmdetection) and [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch) codebases.

We have forked this repository and made changes to build a model that can perform **Semantic Segmentation**. Semantic segmentation involves assigning a class label to each pixel. 

Our work aims to use the video feed from a monocular camera and perform semantic segmentation on this feed. This camera is installed on a drone and segmentation of (bark, bush, road and sky) is a key component required to understand the environment and enable the use autonomous drones for agriculture and forest environments.


We have explored the code and the corresponding paper to understand the modules that are used for semantic segmentation and for instance segmentation and removed the instance segmentation component. We have finetuned the model by changing the hyperparameters and adding data augmentation to get better results. 

Additionally we have added a few layers in the classifier layer to accomodate the use of the pretrained weights provided by the authors and to train on our dataset. 

We have trained our models on the [Freiburg Forest Dataset](http://deepscene.cs.uni-freiburg.de/)  (tree, vegetation, grass, road, sky, obstacle) and [FinnWoodLands](https://github.com/juanb09111/FinnForest) datasets which is a subset of the finnwood forest dataset and is labeled to identify(bark, sky, ground).

In the future we aim to train the dataset on the entire finnWood dataset.

## System Requirements
* Linux 
* Python 3.7
* PyTorch 1.7
* CUDA 11.2
* GCC 7 or 8

The original paper used CUDA 10.2, but we have used the Nvidia A6000 GPU for training which required the use of CUDA 11.2 or above.

## Upgrading Deprecated Modules

The author's code didn't have support for CUDA 11.2 hence, we had to remove legacy code, libraries and use newer versions that support the gpu drivers.

The library that caused the deprecation error is **#include<THC/thc.h>** and other .h files that come under the THC library. We replaced this library with **#include<Aten/Aten.h>** and torch library. This library has the updated functions that were present in the deprecated version, so we replaced the deprecated functions with the newer functions. This allowed us to run the model on our system requirements and available infrastructure.

## Installation
a. Create a conda virtual environment from the modified env.yml file and activate it.
```shell
git clone https://github.com/KalyanRam1234/EfficientPS_Semantic_Segmentation.git
cd EfficientPS
conda env create -n efficientPS_env --file=env.yml
conda activate efficientPS_env
```
b. Install all other dependencies using pip:
```bash
pip install -r requirements.txt
```
c. Install EfficientNet implementation
```bash
cd efficientNet
python setup.py develop
```
d. Install EfficientPS implementation
```bash
cd ..
python setup.py develop
```

If the above steps run successfully then the setup of the code is complete. Now the Dataset needs to be prepared in the format that the dataloader can accept. 

## Prepare datasets

### CityScapes Dataset Setup [Provided By Authors]

It is recommended to symlink the dataset root to `$EfficientPS/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
EfficientPS
├── mmdet
├── tools
├── configs
└── data
    └── cityscapes
        ├── annotations
        ├── train
        ├── val
        ├── stuffthingmaps
        ├── cityscapes_panoptic_val.json
        └── cityscapes_panoptic_val
```
The cityscapes annotations have to be converted into the aforementioned format using
`tools/convert_datasets/cityscapes.py`:
```shell
python tools/convert_cityscapes.py ROOT_DIRECTORY_OF_CITYSCAPES ./data/cityscapes/
cd ..
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/cityscapesscripts/preparation
python createPanopticImgs.py --dataset-folder path_to_cityscapes_gtFine_folder --output-folder ../../../EfficientPS/data/cityscapes --set-names val
```

### Running On Custom Datasets

To run on different datasets, we present a set of instructions to format the dataset to the directory structure, makes masks,etc specified above and fit the dataloader.

All the codes for this section are in the tools directory

1. Make a copy of the [labels.py](/tools/labels.py). Here specify all the classes that are present in the dataset along with its color.

2. Make a copy of the [convert_freiburg.py](/tools/convert_freiburg.py) file. Then import the labels file created in step 1 and use appropriate names for the output files.

3. Ensure that the folder with the raw data is formated as training/images and training/annotations, similarly for validation . Then run the command below.
```
python tools/<Name of convert file> ROOT_DIRECTORY_OF_DATASET OUTPUT_DIRECTORY
```

4. Now the data is formatted as shown in the diagram. We now need to create the DataLoader, to do so copy the [finnforest.py](/mmdet/datasets/finnforest.py) file and here add all the classes that are present in your dataset to the **CLASSES** variable.

5. Import and add this dataloader file to the [__init__.py](/mmdet/datasets/__init__.py) file. We then go to the [config](/configs/efficientPS_singlegpu_sample.py) file present in the configs folder and replace the values of the appropriate fields with the data path and the dataloader to be used.

6. Set the required configurations in the config file, i.e the hyperparameters, etc.

## Training and Evaluation
### Training Procedure

Ensure the metrics is set to segm. The config file we used makes use of the efficientNet_b5 backbone and a 2-way-FPN. Following which only the semantic head is present where the number of classes, etc needs to be adjusted based on dataset. We used this configuration to take advantage of the pretrained cityscapes weights where we did a simple network surgery to remove the panoptic layer states while loading the model.

Train with a single GPU:
```
python tools/train.py efficientPS_singlegpu_sample.py --work_dir work_dirs/checkpoints --validate 
```
Train with multiple GPUS:
```
./tools/dist_train.sh efficientPS_multigpu_sample.py ${GPU_NUM} --work_dir work_dirs/checkpoints --validate 
```
* --resume_from ${CHECKPOINT_FILE}: Resume from a previous checkpoint file.

### Evaluation Procedure

We have added code to calculate the IoU and DiceLoss scores during training and evaluation. They are the standard metrics used to evaluate the performance of a semantic segmentation model.

Test with a single GPU:
```
python tools/test.py efficientPS_singlegpu_sample.py ${CHECKPOINT_FILE} --eval segm
```
Test with multiple GPUS:
```
./tools/dist_test.sh efficientPS_multigpu_sample.py ${CHECKPOINT_FILE} ${GPU_NUM} --eval segm
```

## Pre-Trained Models

### Models By Authors

| Dataset   |  Model | PQ |
|-----------|:-----------------:|--------------|
| Cityscapes| [Download](https://www.dropbox.com/s/zihqct9zum8eq66/efficientPS_cityscapes.zip?dl=0) | 64.4 |
|    KITTI  | [Download](https://www.dropbox.com/s/4z3qiaew8qq7y8n/efficientPS_kitti.zip?dl=0) | 42.5| 

### Our Models

| Dataset | Model |
| ------- |-------|

## WorkFlow

Checkout this readme for more details on our work : [WorkFlow](/workflow.md) 

## Additional Notes:

To make predictions on the semantic segmentation model, we have modified the cityscapes_save_predictions file to get the semantic mask.

   * tool/cityscapes_inference.py: saves predictions in the official cityscapes panoptic format.
   * tool/cityscapes_save_predictions.py: saves color visualizations.

Example: 
```
python tools/cityscapes_save_predictions.py CONFIGS_PATH PTH_MODEL_PATH INPUT_DATA_DIRECTORY OUTPUT_DIRECTORY
```
Note that input directory should have the images in a subdirectories.

## Acknowledgements
The authors have used utility functions from other open-source projects. We especially thank the authors of:
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)
- [seamseg](https://github.com/mapillary/seamseg.git)


## Contacts

### Our Team

* [Haveli UAVs](https://www.haveliuavs.com/)
* [Viswanath G](https://sites.google.com/view/viswanathiiitb/home)
* [Siddharth Chauhan](https://github.com/SiddharthChauhan303)
* [Munagala Kalyan Ram](https://github.com/KalyanRam1234)
* [Prateeth Rao](https://github.com/Prateeth8)


### Original Authors
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
* [Rohit Mohan](https://github.com/mohan1914)


