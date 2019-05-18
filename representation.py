import torch
from torch import nn
from torch.nn import functional as F

class Pyramid(nn.Module):
    def __init__(self):
        super(Pyramid, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7+3, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=8, stride=8),
            nn.ReLU()
        )
        

    def forward(self, x, v):
        # Broadcast
        v = v.view(-1, 7, 1, 1).repeat(1, 1, 64, 64)
        r = self.net(torch.cat((v, x), dim=1))
        
        return r
    
class Tower(nn.Module):
    def __init__(self, v_dim):
        super(Tower, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256+v_dim, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256+v_dim, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn8 = nn.BatchNorm2d(256)

    def forward(self, x, v):
        x = x.contiguous().view(-1, 3, 64, 64)
        # Resisual connection
        skip_in  = F.relu(self.bn1(self.conv1(x)))
        skip_out = F.relu(self.bn2(self.conv2(skip_in)))

        r = F.relu(self.bn3(self.conv3(skip_in)))
        r = F.relu(self.bn4(self.conv4(r))) + skip_out

        # Broadcast
        v = v.contiguous().view(-1, v.size(2), 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.bn5(self.conv5(skip_in)))

        r = F.relu(self.bn6(self.conv6(skip_in)))
        r = F.relu(self.bn7(self.conv7(r))) + skip_out
        r = F.relu(self.bn8(self.conv8(r)))

        return r
    
class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256+7, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256+7, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.pool  = nn.AvgPool2d(16)

    def forward(self, x, v):
        # Resisual connection
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out

        # Broadcast
        v = v.view(v.size(0), 7, 1, 1).repeat(1, 1, 16, 16)
        
        # Resisual connection
        # Concatenate
        skip_in = torch.cat((r, v), dim=1)
        skip_out  = F.relu(self.conv5(skip_in))

        r = F.relu(self.conv6(skip_in))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        
        # Pool
        r = self.pool(r)

        return r