import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from models.vit_extractor import FeatureExtractor
from models.dynamic_residual_feature_aggregation_network import DynamicResidualFeatureAggregationNetwork
from models.train_attractor import LearningModelTrainer
import json
import numpy as np
from utils.visualize import visualize_features
from utils.metrics import calculate_intra_class_compactness, calculate_inter_class_distance
#python train_vit.py
def main():
    with open('config/training_config.json', 'r') as f:
        config = json.load(f)

    transform = Compose([
        Resize((224, 224)),  # Adjust image size to match ViT's expected input size
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extractor = FeatureExtractor(pretrained=True).to(device)
    attention_attractor_net = DynamicResidualFeatureAggregationNetwork(num_classes=10, feature_dim=1000).to(device)
    attractor_trainer = LearningModelTrainer(feature_extractor, attention_attractor_net, config)
    optimizer = optim.AdamW([
        {'params': feature_extractor.parameters(), 'lr': config['lr']},
        {'params': attention_attractor_net.parameters(), 'lr': config['meta_lr']}
    ])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    feature_extractor.train()
    attention_attractor_net.train()
    
    optimizer = optim.SGD(feature_extractor.parameters(), lr=config["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["pretrain_epochs"]):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = feature_extractor(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Pretrain Epoch [{epoch+1}/{config['pretrain_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

                
                
    feature_extractor.eval()  # 设置特征提取器为评估模式
    optimizer = optim.AdamW(attention_attractor_net.parameters(), lr=config["meta_lr"])
    criterion = nn.CrossEntropyLoss()
    lambda_reg = config["lambda_reg"]

    for epoch in range(config["attraction_epochs"]):
        total_loss_accumulated = 0  # 累积损失，用于日志
        for j, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 获取特征表示
            with torch.no_grad():  # 不更新特征提取器的梯度
                features = feature_extractor(images)
            
            # 计算吸引子网络的输出和分类损失
            optimizer.zero_grad()
            outputs = attention_attractor_net(features)
            classification_loss = criterion(outputs, labels)
            
            # 计算正则化损失
            reg_loss = attention_attractor_net.regularize_attractors()

            # 计算总损失并进行反向传播
            total_loss = classification_loss + lambda_reg * reg_loss
            total_loss.backward()
            optimizer.step()

            total_loss_accumulated += total_loss.item()

            if (j + 1) % 100 == 0:
                avg_loss = total_loss_accumulated / 100
                print(f"Attraction Epoch [{epoch+1}/{config['attraction_epochs']}], Step [{j+1}/{len(train_loader)}], Avg Loss over last 100 steps: {avg_loss:.4f}")
                total_loss_accumulated = 0  # 重置累积损失

    # Evaluation loop
    feature_extractor.eval()
    attention_attractor_net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = feature_extractor(images)
            outputs = attention_attractor_net(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on the test dataset: {accuracy * 100:.2f}%')

    # Assuming visualize_features and evaluate_metrics are adapted to work with PyTorch Tensors
    # and the current device (e.g., GPU support)
    # Dummy feature and label extraction for visualization and metric evaluation
    features, labels = next(iter(test_loader))
    features, labels = features.to(device), labels.to(device)
    features = feature_extractor(features)
    labels = labels.cpu().numpy()
    features = features.cpu().detach().numpy()

    # Visualization and Metrics Evaluation
    #visualize_features(features, labels)
    compactness = calculate_intra_class_compactness(torch.tensor(features), torch.tensor(labels))
    inter_class_distance = calculate_inter_class_distance(torch.tensor(features), torch.tensor(labels))
    print(f"Class Intra-Compactness: {compactness.item():.4f}")
    print(f"Inter-Class Distance: {inter_class_distance.item():.4f}")

if __name__ == "__main__":
    main()
    