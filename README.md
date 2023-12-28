# EMNIST-Handwritten-Character-Classification
Convolutional Neural Network (CNN) and Multi-Layer Perceptron (MLP) models for handwritten character classification using the EMNIST dataset.

## Introduction
The objective of this project is to accurately classify handwritten digits and characters using two distinct neural network architectures: Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN). The focus is on leveraging the EMNIST dataset, specifically the balanced split, to address classification challenges present in the ByMerge and ByClass datasets. This balanced dataset consists of 112,800 training samples, 18,800 testing samples, across 47 classes.

### Dataset Example Figures
#### Figures With Label Mapping
![Figures With Label Mapping](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/figures%20with%20label%20mapping.png)

#### Figures Without Label Mapping
![Figures Without Label Mapping](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/figures%20without%20label%20mapping.png)

### Architecture Overview
#### MLP Architecture
The MLP model comprises three fully connected hidden layers, facilitating deeper network depth and increased weight parameters. It serves as the fundamental neural network for this classification task.

#### CNN Architecture
The CNN model integrates two convolutional layers, employing multiple filters to extract local features through convolution operations. These layers are followed by activation functions like relu, leaky_relu, or elu, and pooling layers for downsampling the feature maps.

## Hyperparameter Tuning
The project's approach involves a directed search method due to numerous hyperparameters and the dataset's scale. Each parameter underwent stepwise adjustments, optimizing the model's performance iteratively. Parameters explored in the both models include activation functions, optimizers (Adam, SGD, ASGD), learning rate schedulers, dropout, batch normalization, and regularization techniques (l1, l2).

### Best Parameters for CNN Model
| Parameter        | Chosen Value |
|------------------|--------------|
| Activation       | Relu         |
| Optimizer        | Adam         |
| Learning Rate    | StepLr       |
| Dropout          | False        |
| Batch Normalization | True      |
| Regularization   | l2           |

### Best Parameters for MLP Model
| Parameter         | Chosen Value |
|-------------------|--------------|
| Activation        | Relu         |
| Optimizer         | Adam         |
| Learning Rate     | StepLr       |
| Dropout           | True         |
| Batch Normalization | True       |
| Regularization    | None         |

## Training and Validation
### Accuracy and Loss Function Graphs
#### MLP
![MLP train accuracy](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/MLP-train-val-accuracy.png)
![MLP train loss](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/MLP-train-val-loss.png)

#### CNN
![CNN train accuracy and loss](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/CNN-train-val-loss-accuracy.png)

## Model Evaluation
### Performance Metrics
#### MLP vs. CNN
| Model        | Training Loss | Testing Loss | Testing Accuracy |
|--------------|---------------|--------------|------------------|
| MLP          | 0.3933        | 0.3924       | 86.719%          |
| CNN          | 0.337         | 0.369        | 88.281%          |

## Predicted Results and Analysis
Confusion matrix analysis and comparison between MLP and CNN models based on the achieved performance.
### MLP
![MLP Confusion Matrix](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/MLP-confusionmatrix.png)
### CNN
![CNN Confusion Matrix](https://github.com/sacadelmi/EMNIST-Handwritten-Character-Classification/blob/main/CNN-confusionmatrix.png)

### Model Comparison
#### MLP Pros and Cons
- Pros: Simple architecture, quick training for smaller datasets.
- Cons: Limited ability to capture spatial features, prone to overfitting for high-dimensional data.

#### CNN Pros and Cons
- Pros: Proficient in capturing spatial features, and hierarchical feature learning.
- Cons: More complex architecture, requires larger datasets, longer training time.

### Conclusion
The CNN model demonstrates superior performance, especially in scenarios involving spatial relationships and complex data structures, outperforming the MLP architecture in this handwritten character classification task.
