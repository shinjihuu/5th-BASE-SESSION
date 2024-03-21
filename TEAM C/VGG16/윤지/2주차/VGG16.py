
import torch.nn as nn

# conv_2_block

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential( 
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2) # maxpooling 진행할 때 kernel과 stride 2로 진행
    )
    return model

# conv_3_block

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential( # 모듈 순차 정의
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2) # kernel stride padding 모두 동일
    )
    return model

# Define VGG16

class VGG16(nn.Module): # nn.Module을 상속받아 신경망을 구성하는데 필요한 메소드들 사용 가능
    def __init__(self, base_dim, num_classes = 10):
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), # 64
            conv_2_block(base_dim, 2*base_dim), # 128
            conv_3_block(2*base_dim, 4*base_dim), # 256
            conv_3_block(4*base_dim, 8*base_dim), # 512
            conv_3_block(8*base_dim, 8*base_dim) # 512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*3*3, 4096),
            #32x32
            #224x224 -> (8*base_dim*7*7, 4096)
            #96x96 -> (8*base_dim*3*3, 4096)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc_layer(x)
        return x
        