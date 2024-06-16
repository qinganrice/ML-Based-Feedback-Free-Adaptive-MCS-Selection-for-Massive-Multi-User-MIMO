import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.buildcnn import CNNModel


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = CNNModel()
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    ### x_3d [batch_size, sequential, C, H, W]
    
    def forward(self, x_3d):

        hidden = None
        for t in range(x_3d.size(1)):
            # with torch.no_grad():
            x = self.cnn(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x