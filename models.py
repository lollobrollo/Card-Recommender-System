import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class HybridConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()        
        self.encoder_conv = nn.Sequential(
            # Input: (3, 936, 672), not considering batch size
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # -> (16, 468, 336)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # -> (32, 234, 168)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # -> (64, 117, 84)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # -> (128, 58, 42)
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2), # -> (128, 29, 21)
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2), # -> (128, 14, 10)
        )
        self.flattened_size = 128 * 14 * 10
        self.encoder_linear = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.flattened_size, latent_dim),
            nn.ReLU(True)
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            # Input will be (128, 14, 10) after reshaping
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, output_padding=(1,1)), # -> (128, 29, 21)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2), # -> (128, 58, 42)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=(1,0)), # -> (64, 117, 84)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # -> (32, 234, 168)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # -> (16, 468, 336)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2), # -> (3, 936, 672)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        return self.encoder_linear(x)    
        
    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.view(-1, 128, 14, 10)
        return self.decoder_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))


class CardImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        self.transform = transform
        if transform == None:
            self.transform = transforms.Compose([
            transforms.Resize((936, 672)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = self.transform(Image.open(img_path).convert('RGB'))
        return img

