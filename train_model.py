import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PoseDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name, label in [("fall", 1), ("nonfall", 0)]:
            class_dir = os.path.join(root_dir, label_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path).astype(np.float32)  # (30, 66)
        return torch.tensor(data), torch.tensor(label)

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super(FallDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=(2, 2))
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, 30, 66)
        x = self.pool1(F.relu(self.conv1(x)))  # (B, 16, 15, 33)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 32, 7, 16)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

data_dir = "pose_data"
batch_size = 16
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PoseDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = FallDetectionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = []
for epoch in range(epochs):
    loss, acc = train(model, dataloader, criterion, optimizer, device)
    history.append((loss, acc))
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")

torch.save(model.state_dict(), "fall_cnn_model.pth")
