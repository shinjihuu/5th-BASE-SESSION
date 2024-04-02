import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset import dataset
from unet import UNet
from tqdm.notebook import tqdm

def save(ckpt_dir,model):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_name = f"/epoch{epochs}.pth"
    torch.save(model.state_dict(), ckpt_dir + model_name)
    


###################################
lr = 1e-2
batch_size = 8
epochs = 20

data_loader = DataLoader(dataset, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3,n_classes=30,bilinear=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

ckpt_dir = os.path.join('./ckpt','unet')

def train():
    print("-----------start train----------")
    step_losses = []
    epoch_losses = []
    for epoch in range(epochs):
        print(f"****** epoch {epoch} *****")
        epoch_loss = 0

        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device) # torch.Size([4, 3, 256, 256]) torch.Size([4, 256, 256])
            optimizer.zero_grad()
            Y_pred = model(X) # torch.Size([4, 10, 256, 256])
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())

        epoch_losses.append(epoch_loss/len(data_loader))
        print(f"epoch {epoch} loss: ", epoch_losses[epoch])
       

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[0].set_title('Step Losses')
    axes[1].plot(epoch_losses)
    axes[1].set_title('Epoch Losses')
    fig.savefig('loss.png')

    save(ckpt_dir = ckpt_dir,model = model)

if __name__ == '__main__':
    train()