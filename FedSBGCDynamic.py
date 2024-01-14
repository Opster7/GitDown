import torch
import torch.nn as nn
import torch.optim as optim
from SBGM import Classifier, ScoreNet
from torch.utils.data import DataLoader,Dataset
from sdenoise import marginal_prob_std_fn, device
import os
import pickle
import logging
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Configure logging
log_file = './SBGM/FedDynamic/cifar10/feature/SBGM-feature-training.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
writer = SummaryWriter('./SBGM/FedDynamic/cifar10/feature/')

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

def main():
    score_net = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_net = score_net.to(device='cuda:0')
    global_model = Classifier(score_net, num_classes=10)
    global_model = global_model.to(device='cuda:0')

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)  # 使用Adam优化器

    batch_size = 32
    data_dir = "./data/cifar100-c-feature-only/"

    num_epochs = 50
    test_data_path = "./data/cifar100-c-feature-only/test-1.pkl"
    test_dataset = CustomDataset(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load from checkpoint if exists
    start_epoch = 0  # Initialize start epoch
    checkpoint_path = "./SBGMDeeper/cifar100/feature/checkpoint_latest.pth"  # Path to your latest checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        global_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")

    client_weights = []  # This will store the weight for each client's update

    for epoch in range(start_epoch, num_epochs):
        global_model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        global_params = None
        total_num_updates = 0  # Track the total number of updates from all clients

        for client_id in range(100):
            data_path = os.path.join(data_dir, f"{client_id}.pkl")
            if not os.path.exists(data_path):
                continue
            print(f"Loading data from {data_path}")

            client_dataset = CustomDataset(data_path)
            client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            client_optimizer = optim.Adam(global_model.parameters(), lr=0.001)

            local_updates = 0  # Track the number of updates for this client

            for local_epoch in range(10):
                for inputs, labels in client_dataloader:
                    client_optimizer.zero_grad()
                    time_steps = torch.rand(inputs.shape[0], device=device)
                    inputs = inputs.permute(0, 3, 1, 2).to(device=device, dtype=torch.float)
                    labels = labels.to(device=device)
                    outputs = global_model(inputs, time_steps)

                    loss = criterion(outputs, labels)
                    if torch.isnan(loss):
                        print('Loss became NaN during training!')
                        continue
                    loss.backward()
                    client_optimizer.step()

                    total_loss += loss.item()
                    total_samples += inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()

                local_updates += 1

            # Append the weight for this client (number of data points or number of updates)
            client_weight = len(client_dataset)  # or you could use local_updates
            client_weights.append(client_weight)
            total_num_updates += client_weight

            if global_params is None:
                global_params = [param.data.clone() * client_weight for param in global_model.parameters()]
            else:
                for global_param, client_param in zip(global_params, global_model.parameters()):
                    global_param += client_param.data * client_weight

        # Aggregate updates
        with torch.no_grad():
            for param, global_param in zip(global_model.parameters(), global_params):
                param.data = global_param / total_num_updates

        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.2f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.2f}")
        writer.add_scalar('Training Loss', total_loss / total_samples, epoch)

        training_accuracy = 100 * correct_predictions / total_samples
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {training_accuracy:.2f}%")
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {training_accuracy:.2f}%")
        writer.add_scalar('Training Accuracy', training_accuracy, epoch)

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:  # Save the last epoch as well
            checkpoint_path = f"./SBGMDeeper/FedDynamic/cifar100/feature/checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': global_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
            logging.info(f"Saved checkpoint at {checkpoint_path}")
            print(f"Saved checkpoint at {checkpoint_path}")

        global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                time_steps = torch.rand(inputs.shape[0], device=device)
                outputs = global_model(inputs, time_steps)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
    writer.close()


if __name__ == "__main__":
    main()

