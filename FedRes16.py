import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import copy
import os
import pickle
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_file = './ResNet/FedAvg/cifar10/feature/Res-feature-training.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
writer = SummaryWriter(log_dir='./Res16/FedAvg2/cifar100/feature/')


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as file:
            self.data = pickle.load(file)
        self.transform = transform

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image_data = self.data["images"][idx]
        target = self.data["labels"][idx]
        if self.transform:
            image_data = self.transform(image_data)
        target = torch.tensor(target, dtype=torch.long)
        return image_data, target


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256 * block.expansion * 2 * 2, num_classes)
        # self.linear = nn.Linear(256 * block.expansion * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.reshape(out.size(0), -1) # out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet16(num_classes=10):  # 默认参数设置为10个类
    return ResNet(BasicBlock, [6, 6, 6], num_classes=num_classes)

def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct, total

def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.permute(0, 3, 1, 2).to(device).float()  # data shape is (batch_size, channels, height, width)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        c, t = calculate_accuracy(output, target)
        correct += c
        total += t

    train_loss = train_loss / len(train_loader)
    train_accuracy = (100. * correct) / total
    return train_loss, train_accuracy

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.permute(0, 3, 1, 2).to(device).float()  # data shape is (batch_size, channels, height, width)
            target = target.to(device)

            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()

            c, t = calculate_accuracy(output, target)
            correct += c
            total += t

    test_loss /= len(test_loader)
    test_accuracy = (correct / total) * 100.0
    return test_loss, test_accuracy

def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients during evaluation
        for data, targets in dataloader:
            data = data.permute(0, 3, 1, 2).to(device).float()
            targets = targets.to(device)
            outputs = model(data)
            c, t = calculate_accuracy(outputs, targets)
            correct += c
            total += t
    accuracy = 100 * correct / total
    return accuracy

# Update the federated_averaging function to handle models directly
def federated_averaging(global_model, client_models, client_weights):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_weights[i][key].float() for i in range(len(client_weights))], 0).mean(0)
    global_model.load_state_dict(global_dict)

    # Update client models with the new global model weights
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    return global_model

def main():
    # 全局模型
    global_model = ResNet16(num_classes=100)
    global_model.to(device)
    global_optimizer = optim.Adam(global_model.parameters(), lr=1e-3)

    num_clients = 100 # 客户端数量
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    client_optimizers = [optim.Adam(client_models[i].parameters(), lr=1e-3) for i in range(num_clients)]

    # 数据加载和预处理
    data_dir = "./data/cifar10-c-feature-only/"
    batch_size = 32
    num_epochs = 50

    test_data_path = "./data/cifar10-c-feature-only/test-1.pkl"
    test_dataset = CustomDataset(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    start_epoch = 0
    checkpoint_path = "./ResNet/FedAvg/cifar10/feature/checkpoint_epoch_20.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        global_model.load_state_dict(checkpoint['model_state_dict'])
        global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        global_model.train()
        client_losses = []
        client_accuracies = []

        # Prepare to collect client model weights
        client_weights = []

        # Training each client model
        for client_id in range(num_clients):
            local_model = client_models[client_id]
            local_optimizer = client_optimizers[client_id]
            data_path = os.path.join(data_dir, f"{client_id}.pkl")

            # Ensure the data path exists
            if not os.path.exists(data_path):
                continue

            # Load the client's data
            client_dataset = CustomDataset(data_path)
            client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

            # Train the local model
            client_loss, _ = train(local_model, client_dataloader, local_optimizer, device)
            client_losses.append(client_loss)
            client_accuracy = evaluate(local_model, test_dataloader,
                                       device)  # Assuming you have a validation_dataloader
            client_accuracies.append(client_accuracy)

            # Collect local model weights
            client_weights.append(copy.deepcopy(local_model.state_dict()))

        # Perform federated averaging
        global_model = federated_averaging(global_model, client_models, client_weights)

        # Average the losses for logging purposes
        avg_loss = sum(client_losses) / len(client_losses)
        avg_accuracy = sum(client_accuracies) / len(client_accuracies)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}, Average Training Accuracy: {avg_accuracy}%")
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_loss:.2f}, Average Training Accuracy: {avg_accuracy:.2f}%")

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = f"./Res16/FedAvg/cifar10/feature/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': global_model.state_dict(),
                'optimizer_state_dict': global_optimizer.state_dict()
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at {checkpoint_path}")
            print(f"Saved checkpoint at {checkpoint_path}")

        test_loss, test_accuracy = test(global_model, test_dataloader, device)
        test_accuracy_percentage = test_accuracy

        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy_percentage:.2f}%")
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy_percentage:.2f}%")

    writer.close()

if __name__ == "__main__":
    main()
