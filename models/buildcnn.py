import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define layers and operations in the __init__ method
        self.conv1 = nn.Conv2d(2, 16, (3,2), padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 16, (3,2), padding=1, stride=1)
        self.conv3 = nn.Conv2d(16, 32, (3,2), padding=1, stride=1)
        self.conv4 = nn.Conv2d(32, 32, (3,2), padding=1, stride=1)
        self.conv5 = nn.Conv2d(32, 64, (3,2), padding=1, stride=1)
        self.conv6 = nn.Conv2d(64, 64, (3,2), padding=1, stride=1)
        self.conv7 = nn.Conv2d(64, 64, (3,2), padding=1, stride=1)

        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.fc = nn.Linear(436224, 300)

        
    def dense_block(self, dense_in, conv1, conv2, bn):
        hidden_c = F.relu(conv1(dense_in))
        hidden_v = F.relu(conv2(hidden_c))

        for i in range(2):
            hidden_c = torch.cat((hidden_c, hidden_v), axis=-1)
            hidden_v = F.relu(conv2(hidden_c))

        hidden_v = bn(hidden_v)
        
        return hidden_v

    def forward(self, x):
        x = self.dense_block(x, self.conv1, self.conv2, self.bn1)
        x = self.pool(x)
        x = self.dense_block(x, self.conv3, self.conv4, self.bn2)
        x = self.pool(x)
        x = self.dense_block(x, self.conv5, self.conv6, self.bn3)
        x = self.dense_block(x, self.conv6, self.conv7, self.bn4)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print(x.size(1))
        x = self.fc(x)
        # x = F.relu(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        return x


