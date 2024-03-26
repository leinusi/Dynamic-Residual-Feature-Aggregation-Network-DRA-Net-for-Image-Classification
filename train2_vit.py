import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import argparse
import os
from models.vit_extractor import FeatureExtractor
from models.dynamic_residual_feature_aggregation_network import DynamicResidualFeatureAggregationNetwork
from models.train_attractor import LearningModelTrainer
#python train2_vit.py --data_dir data/eyes --batch_size 4 --lr 0.001 --meta_lr 0.0001 --pretrain_epochs 5 --attraction_epochs 3 --num_classes 6


def load_data(data_dir, transform, split_ratio=0.7):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    train_dataset, test_dataset = load_data(args.data_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_extractor = FeatureExtractor(pretrained=True).to(device)
    attention_attractor_net = DynamicResidualFeatureAggregationNetwork(num_classes=10, feature_dim=1000).to(device)
    attractor_trainer = LearningModelTrainer(feature_extractor, attention_attractor_net, args)
    optimizer = optim.AdamW([
        {'params': feature_extractor.parameters(), 'lr': args.lr},
        {'params': attention_attractor_net.parameters(), 'lr': args.meta_lr}
    ])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    feature_extractor.train()
    attention_attractor_net.train()
    
    optimizer = optim.SGD(feature_extractor.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.pretrain_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = feature_extractor(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Pretrain Epoch [{epoch+1}/{args.pretrain_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

                
                
    feature_extractor.eval()  # 设置特征提取器为评估模式
    optimizer = optim.AdamW(attention_attractor_net.parameters(), lr=args.meta_lr)
    criterion = nn.CrossEntropyLoss()
    lambda_reg = args.lambda_reg

    for epoch in range(args.attraction_epochs):
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
                print(f"Attraction Epoch [{epoch+1}/{args.attraction_epochs}], Step [{j+1}/{len(train_loader)}], Avg Loss over last 100 steps: {avg_loss:.4f}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model with command line parameters.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the feature extractor')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate for the DRFAN')
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--attraction_epochs', type=int, default=3, help='Number of epochs to train for')

    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--lambda_reg', type=int, default=0.001, help='lambda_reg')

    args = parser.parse_args()

    main(args)
