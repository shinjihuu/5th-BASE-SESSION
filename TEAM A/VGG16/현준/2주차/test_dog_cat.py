import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import DogCatDataset

from Vgg_new import VGG16

import argparse

# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model = VGG16(init_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test(model, test_loader):
    correct, total = 0, 0

    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y = label.to(device)

            output = model.forward(x)
            _, output_idx = torch.max(output,1)

            total += label.size(0)
            correct += (output_idx == y).sum().float()
        print(f'Accuracy of Test Data :{100*correct/total}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VGG16 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.pth)")
    
    args = parser.parse_args()
    
    # 모델 로드
    model = load_model(args.model_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_dataset = DogCatDataset(data_dir='dog_cat/test1', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 테스트 실행
    test(model, test_loader)