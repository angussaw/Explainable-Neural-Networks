# Explainable-Neural-Networks for pneumonia detection Gradient-weighted Class Activation Mapping (Grad-CAM)





## Introduction

The objective of this project is to develop a deep learning model to classify X-ray images into one of the three classes: normal, bacteria and virus for pneumonia detection

To understand the classifications made by the deep learning model and enhance explainability its black box nature, the visual explaination algorithm Gradient-weighted Class Activation Mapping (Grad-CAM) generates heatmaps that are superimposed on the test images to provide visual explainability on the model. The heatmap highlights the important regions in the X-ray image for predicting the classes.
## Data

```bash
├── data
│   ├── test
│   │   ├── BACTERIA
│   │   └── NORMAL
│   │   └── VIRUS
│   ├── train
│   │   ├── BACTERIA
│   │   ├── NORMAL
│   │   └── VIRUS
```

**Percentage of classes in train and test set**

|         dataset|   train|    test|
|----------------|--------|--------|
| pct_of_bacteria|  48.581|  38.782|
|   pct_of_normal|  25.709|  37.500|
|    pct_of_virus|  25.709|  23.718|

**Number of classes in train and test set**

|            dataset|  train|  test|
|-------------------|-------|------|
| number_of_bacteria|   2534|   242|
|   number_of_normal|   1341|   234|
|    number_of_virus|   1341|   148|

There are a total of 5216 and 624 images in the train and test set respectively. The train set is further split to obtain a validation set for model training. Below are sample images for each of the classes:

![image info](./images/samples.png)



## Model

Transfer learining is used to build the model. The model utilizes the pre-trained weights from MobileNetV2 as a starting point. 
The layers in the base model is frozen to be used as a feature extractor.

The architecture of the model consists of:
- Input layer
- RandomFlip and Random Rotation layers for data augmentation (Note: Data augmentation is inactive at test time)
- Rescaling layer for standardization
- MobileNetV2 layers previously trained on the 'imagenet' dataset. (Layers are frozen for transfer learning)
- Global max pooling layer
- Dropout layer
- Number of dense layers with ReLu activation function
- Output layer of 3 classes with softmax activation


## Evaluation of baseline model

**Baseline model training parameters**
|   |   |
|---|---|
|   |   |
|   |   |
|   |   |
|   |   |
|   |   |

**Confusion Matrix of test data**

![image info](./mlartifacts/140308137141938706/cab1e769583c4ce9a4a54141e0d79f57/artifacts/Confusion_Matrix%20(test%20data).png)

**Test data metrics for each class**

![image info](./mlartifacts/140308137141938706/cab1e769583c4ce9a4a54141e0d79f57/artifacts/Test%20metrics%20for%20each%20class.png)


### What the model did well

![image info](./mlartifacts/140308137141938706/cab1e769583c4ce9a4a54141e0d79f57/artifacts/Sample%20Grad-CAM%20visualizations%20on%20test%20data%20(NORMAL-True%20Positives).png)


### What the model didn't do well

![image info](./mlartifacts/140308137141938706/cab1e769583c4ce9a4a54141e0d79f57/artifacts/Sample%20Grad-CAM%20visualizations%20on%20test%20data%20(VIRUS-False%20Negatives).png)

![image info](./mlartifacts/140308137141938706/cab1e769583c4ce9a4a54141e0d79f57/artifacts/Sample%20Grad-CAM%20visualizations%20on%20test%20data%20(BACTERIA-False%20Positives).png)