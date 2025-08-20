import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


# - - - - - - - - - - - - - - - - - Models for card image representation - - - - - - - - - - - - - - - - - 

class HybridConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            # Input: (N, 3, 936, 672), not considering batch size
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


# - - - - - - - - - - - - - - - - - Models for contextual preference ranking pipeline - - - - - - - - - - - - - - - - - 


class CardEncoder_v1(nn.Module):
    def __init__(self, card_dim=1446, hidden_dim=1024, out_dim=512, p_drop=0.3):
        super().__init__()
    
        self.MLP = nn.Sequential(
            nn.Linear(card_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop)
        )

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Assuming x of size (n_batch, input_dim)
        x = self.MLP(x)
        x = self.out(x)
        return F.normalize(x, p=2, dim=1)


class DeckEncoder_v1(nn.Module):
    """
    Treats the deck as a sequence of cards -> uses Conv1d to extract features
    """
    def __init__(self, card_dim=1446, out_dim=512):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=card_dim, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ELU(True), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ELU(True), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ELU(True), nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(True),
            nn.AdaptiveMaxPool1d(1)
        )

        self.out = nn.Sequential(
            nn.Flatten(start_dim=1),  # (Batch, 64, 1) -> (Batch, 64)
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # Permute the dimensions to match Conv1d's expectation.
        # (Batch, Num Cards, Card Dim) -> (Batch, Card Dim, Num Cards)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.out(x)
        return F.normalize(x, p=2, dim=1)


class DeckEncoder_v2(nn.Module):
    """
    Treats the deck as an image -> uses Conv2d to extract features
    """
    def __init__(self, output_dim=512):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ELU(True), nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ELU(True), nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ELU(True), nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ELU(True), nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ELU(True), nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ELU(True), nn.MaxPool2d(kernel_size=(2,2), stride=2),
        )

        self.out = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
    # x is expected to be (batch_size, num_cards, card_vector_dim)
    # Using unsqueeze to artificially create a color channel (deck as a greyscale image)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.out(x)
        return F.normalize(x, p=2, dim=1)


class SiameseHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, out_dim=2, p_drop=0.3):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(True), nn.Dropout(p_drop)
        )

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # Expected input: (batch_size, input_dim)
        x = self.MLP(x)
        x = self.out(x)
        return torch.tanh(x)


class PipelineCPR(nn.Module):
    def __init__(self, card_dim, card_hidden_dim, embed_dim=512, out_dim=2):
        super().__init__()

        self.card_encoder = CardEncoder_v1(card_dim=card_dim, hidden_dim=card_hidden_dim, out_dim=embed_dim)
        self.deck_encoder = DeckEncoder_v1(card_dim=card_dim, out_dim=embed_dim)
        self.siamese_head = SiameseHead(input_dim=embed_dim, hidden_dim=embed_dim, out_dim=out_dim)

    def forward(self, a, p, n):
        anchor = self.siamese_head(self.deck_encoder(a))
        positive = self.siamese_head(self.card_encoder(p))
        negative = self.siamese_head(self.card_encoder(n))

        return anchor, positive, negative


class DatasetCPR(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
