# Read Scene Understanding Using ModifiedUNet Model

This project contains two experiments for semantic segmentation using a Modified Version of the U-Net model. The first experiment is semantic segmentation for one class, like cars, and the second experiment considers multiple classes. The experiments run on the Pytorch platform and Python 3.7. Mapillary Vistas dataset is used during the experiments.  

## Required libraries
- [Numpy](https://numpy.org/install/)
- [Pytorch](https://pytorch.org/)
- [OpenCV-python](https://pypi.org/project/opencv-python/)
- [Albumentations](https://albumentations.ai/)
- [torchvision](https://pytorch.org/)
- [tqdm](https://tqdm.github.io/)

## Important files
- `parameters.py`: All project parameters are defined in this file, including input and output paths.
- `dataset.py`: A helper class for training takes images and targets dataset, then provides pair-batches of the dataset during the training process.
- `models.py`: This file contains an implementation of the U-Net modified version.
- `prediction.py`: This file applies the saved trained semantic segmentation model to a video. It generates a new video with the segmentation results. 
- `imageprocessing.py`: This file generates semantic segmentation masks given a label segmentation image. The masks are provided in separated folders based on the class name.

## Experiment (1): One-Class Semantic Segmentation.
In this experiment, a semantic segmentation for one class is done by approaching cars as a target class. The training is only applied on 5K~8K images and 1.2~1.5K images as a validation dataset. The experiement code is provided in `vehicle-segmentation` folder. The pixel accuracy reachs to 87%. The trained model is applied on a video recorded in Dubai, UAE.
![alt text](https://user-images.githubusercontent.com/20774864/121232074-41417c80-c8a2-11eb-9c91-f891974ea69f.png)

## Experiment (2): Multi-Classes Semantic Segmentation.
In this experiment, we consider five classes: road, sidewalk, curb, crosswalk, and lane line. The experiement code is provided in `sidewalk-segmentation` folder. The accuracy reaches 95% pixel accuracy. 
![alt text](https://user-images.githubusercontent.com/20774864/147135092-ecad45f3-a57a-4abe-a918-adecd7c193c2.PNG)


