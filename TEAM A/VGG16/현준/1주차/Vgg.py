import torch.nn as nn
import torch.nn.functional as F

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

class VGG16(nn.Module):
    def __init__(self, init_dim, num_classes=10):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,init_dim),
            conv_2_block(init_dim, init_dim*2),
            conv_3_block(init_dim*2, init_dim*4),
            conv_3_block(init_dim*4, init_dim*8),
            conv_3_block(init_dim*8, init_dim*8)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(init_dim*8*1*1, 4096),
            # if dataset = Imagenet
            # nn.Linear(init_dim*8*7*7, 4096)
            nn.ReLU(True),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000,num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layer(x)
        x_prob = F.softmax(x,dim=1)
        return x_prob