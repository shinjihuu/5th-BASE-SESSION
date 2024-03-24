import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): 데이터가 위치한 디렉토리 경로.
            transform (callable, optional): 샘플에 적용될 Optional transform.
        """
        self.data_dir = data_dir
        self.transform = transform
        # 데이터 디렉토리 내 모든 파일 이름을 리스트로 저장
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_name)
        label = 1 if 'dog' in img_name.split('/')[-1] else 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, label