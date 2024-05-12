## Using pretrained weights
Initially we started using pre-trained weights of the model on CityScapes as well as KITTI dataset by removing the semantic segmentation head entirely. Since the classes of the above datasets and our needs were different, the inference was not that good. 
![Original Image](readme/images/original.png)
![KITTI Pre-Trained Weights](/readme/images/kitti.png)
![Cityscapes Pre-Trained Weights](/readme/images/cityscapes.jpg)
## Separating Semantic head from the complete model
This repository solely focuses on the semantic segmentation aspect of the EfficientPS architecture which is primarily used for Panoptic Segmentation. We have commented all the auxillary code not useful for this part, which is very useful when it comes to traing and inference on new images. We observed a 50% reduction in training (1 epoch) as well as inference time for each new image. 
[EfficientPS paper](https://arxiv.org/pdf/2004.02307)
Referring this paper, we have completely removed the instance segmentation head, which consists of the RPN ( Region Proposal Network), Bounding box predictions and the entire Panoptic Fusion Module. 
## Training
We trained our model from scratch on the frieburg forest dataset, but the dataset size as well as the model parameters were huge. This led to a slower convergence rate and poor results. (Insert Results)
## Setting up the dataset 
For training the model on various datasets like freiburg forest, Finnwood Forest, Tartan Air, CityScapes, KITTI, etc we needed to convert all of the datasets into a pre-specified COCO format, whose python .py files can be found in the /tools directory.
## Dataset Description

### FinnWoodLands Forest Dataset
- **Description**: Dataset of forest images for computer vision and forestry research.
- **Classes**: Various tree types, vegetation, terrain features, and potential wildlife or human objects.
- **Total Images**: Over 5,000 Images
- **Image Dimensions**: Varies, often high-resolution suitable for detailed forest analysis.
- **Link**: [Link](https://github.com/juanb09111/FinnForest)

### Freiburg Forest Dataset
- **Description**: Forest image dataset captured around Freiburg, Germany.
- **Classes**: Tree species, undergrowth, forest paths, rocks, water bodies, and urban elements.
- **Total Images**: Over 15,000 Images.
- **Image Dimensions**: Typically high-resolution for outdoor scene understanding.
- **Link**: [Link](https://paperswithcode.com/dataset/freiburg-forest)

### Cityscapes Dataset
- **Description**: Urban street scene dataset for semantic understanding and autonomous driving research.
- **Classes**: 30 classes including road, sidewalk, buildings, vehicles (car, bus, truck), pedestrians, traffic signs, vegetation, sky, and more.
- **Total Images**: More than 5,000 images across different cities.
- **Image Dimensions**: High-resolution (1024x2048 pixels) suitable for detailed urban scene analysis.
- **Link**: [Link](https://www.cityscapes-dataset.com/)
### KITTI Dataset
- **Description**: Dataset for autonomous driving research with vehicle-mounted camera images.
- **Classes**: 8 classes including car, van, truck, pedestrian, cyclist, tram, and miscellaneous.
- **Total Images**: More than 14,000 images captured under various driving conditions.
- **Image Dimensions**: Typically 1242x375 pixels, suitable for object detection and scene understanding in driving scenarios.
- **Link**: [Link](https://paperswithcode.com/dataset/kitti/)
## HyperParameter Tuning
We used a grid searching technique to find out the best set of hyperparametrs like optimizer, learning rate, image augmentations in the constrained amount of time and resources avaialable to us. The best set of hyper-parameters are already preloaded and ready to use on the config files present for each model and dataset in the /configs directory. The single or multiple GPU usage facility is still available to anyone having varying amount of computational resources.
## Metrics and Loss Functions
Since the original Panoptic Segmentation metrics like Panoptic Quality (PQ), etc were not useful to us anymore, we created new metrics specially made for semantic segmentation like IoU, DICE-score, Precision, Recall, Accuracy, etc.
We also used a custom loss function loss function like DICE-loss which gave better results than the vanilla Binary-Cross Entropy loss.

## Training using Pre-Trained weights
We used the provided cityscapes and kitti datasets pretrained weights as features extractors on our new dataset (freiburg forest) . We added a new trainable softmax layer at the model output to adjust the model to the number of classes in our dataset. The convergence rate was pretty fast as well as the results were also satisfactory.

