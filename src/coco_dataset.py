import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image

class CocoDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.filenames = os.listdir(root)
        self.transform_pipeline = transforms.Compose([transforms.Resize((224, 224)),
                                                      transforms.ToTensor()])

    def __getitem__(self, index):
        image_filename = self.filenames[index]
        image = Image.open(os.path.join(self.root, image_filename))
        return self.transform_pipeline(image)
    
    def __len__(self):
        return len(self.filenames)
