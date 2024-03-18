import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vgg16 import vgg
from data import test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vgg(dim=64).to(device)
model.load_state_dict(torch.load('./trained_model/vgg16.pth'))


correct = 0
total = 0

model.eval()

with torch.no_grad():
    for image,label in test_loader:

        x = image.to(device)
        y= label.to(device)

        output = model.forward(x)
        _,output_index = torch.max(output,1)


        total += label.size(0)
        correct += (output_index == y).sum().float()

    print("Accuracy of Test Data: {}%".format(100*correct/total))