# Dynamic Residual Feature Aggregation Network (DRA-Net) for Image Classification

This repository contains the implementation of a deep learning model for image classification called Dynamic Residual Feature Aggregation Network (DRA-Net). The model is designed to improve the discriminative power and robustness of feature representations by introducing learnable attractors and attention mechanisms.

## Model Architecture

The DRA-Net consists of two main components:

1. **Feature Extractor**: A modified ResNet-18 model is used as the feature extractor. It can be initialized with pre-trained weights or trained from scratch. The last layer can be removed to extract features.

2. **Dynamic Residual Feature Aggregation Network**: This module performs feature aggregation using learnable attractors.
   - Attractors are learned to capture discriminative features for each class.
   - An Adaptive Weights Network generates attractor weights for each sample.
   - A regularization term encourages diversity among attractors by minimizing their pairwise distances.
   - The aggregated features are added to the original features through a residual connection, followed by an MLP for classification.
   - Functions are provided to compute intra-class compactness and inter-class distance.

## Training Process

The training process consists of two stages:

1. **Feature Extractor Pre-training**: The feature extractor is pre-trained using the classification loss.

2. **Attractor Network Training**: The feature extractor is fixed, and the attention attractor network is trained using a weighted sum of the classification loss and the attractor regularization loss.

The model is evaluated on the test set for accuracy, and the feature representations are visualized using t-SNE. Intra-class compactness and inter-class distance metrics are also computed on the test set.

## Dataset

The code uses the CIFAR-10 dataset for training and evaluation. The dataset is automatically downloaded if not found in the specified directory.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/your-username/DRA-Net.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the code:
   ```
   python main.py
   ```

The code will start the training process, evaluate the model on the test set, visualize the features using t-SNE, and compute the intra-class compactness and inter-class distance metrics.

## Results

The model achieves high accuracy on the CIFAR-10 dataset and demonstrates improved feature representation quality through visualization and metric evaluation.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The ResNet-18 model is used as the feature extractor, which was introduced in the paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
- The CIFAR-10 dataset is used for training and evaluation, which was introduced in the paper "Learning Multiple Layers of Features from Tiny Images" by Alex Krizhevsky.
