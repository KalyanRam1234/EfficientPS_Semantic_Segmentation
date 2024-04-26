Notes: (Similar to Readme but with a few tweaks) 

Setup should have the environment with the versions specified in the readme, i.e 
Linux
Python 3.7
PyTorch 1.7
CUDA 10.2
GCC 7 or 8
Installation : 

Conda Installation -  https://www.hostinger.in/tutorials/how-to-install-anaconda-on-ubuntu/



git clone https://github.com/DeepSceneSeg/EfficientPS.git
cd EfficientPS
conda env create -n efficientPS_env --file=environment.yml
conda activate efficientPS_env



Ensure Cuda 10.2 and pytorch 1.7 are installed or versions that are compatible with each other - https://pytorch.org/get-started/previous-versions/    (This link should work else use other one)  https://github.com/nerfstudio-project/nerfstudio/issues/739
pip install -r requirements.txt
pip install inplace-abn=1.0.12
pip install mmcv==0.4.3


 EfficientNet implementation
cd efficientNet
python setup.py develop



EfficientPS implementation installation
cd ..
python setup.py develop




For getting output on random images - tool/cityscapes_save_predictions.py 

Example : python tools/cityscapes_save_predictions.py ./configs/efficientPS_singlegpu_sample.py ./model.pth ./test ./output

python tools/freiburg_save_predictions.py ./configs/efficientPS_singlegpu_sample.py ./epoch_81.pth ./forestimages ./output/freiburgtestfinal2

Train : 
python ./tools/train.py ./configs/efficientPS_singlegpu_sample.py --work_dir work_dirs/checkpoints --resume_from work_dirs/checkpoints/epoch_3.pth

Creating coco formatted dataset

python tools/convert_freiburg.py FreiburgForestDataset ./data/freiburg/



 