import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG16
import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

batch_size = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5),(0.5))]
)


test_data = datasets.DatasetFolder(root="./data/skintypes/test", transform=transform)

test_loader = DataLoader(test_data, batch_size=batch_size)

# Train
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./models/VGG16_newdata.pth'))

# evl
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for i, [image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _, output_index = torch.max(output,1)

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))