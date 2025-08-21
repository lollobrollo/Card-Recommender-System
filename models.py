import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
from torch.nn.utils.rnn import pad_sequence

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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Assuming x of size (n_batch, input_dim)
        x = self.MLP(x)
        return F.normalize(x, p=2, dim=1)


class DeckEncoder_v1(nn.Module):
    """
    Treats the deck as a sequence of cards -> uses Conv1d to extract features
    """
    def __init__(self, card_dim=1446, out_dim=512):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=card_dim, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ELU(True), nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ELU(True), nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ELU(True), nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ELU(True),
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ELU(True),
            nn.Conv1d(32, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ELU(True),
            nn.AdaptiveMaxPool1d(1)
        )

        self.out = nn.Sequential(
            nn.Flatten(start_dim=1),  # (Batch, 64, 1) -> (Batch, 64)
            nn.Linear(16, out_dim)
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

    def card_embedding(self, x):
        return self.siamese_head(self.card_encoder(x))

    def deck_embedding(self, x):
        return self.siamese_head(self.deck_encoder(x))

    def forward(self, a, p, n):
        anchor = self.deck_embedding(a)
        positive = self.card_embedding(p)
        negative = self.card_embedding(n)

        return anchor, positive, negative


class TripletEDHDataset(Dataset):
    """
    PyTorch Dataset that generates training triplets on the fly for the PipelineCPR
    """
    def __init__(self, decklists, card_feature_map, anchor_size_range=(50, 95)):
        """
        Args:
            decklists (list): A list of lists containing the oracle_ids of the 100 cards in a deck
            card_feature_map (dict): A dictionary mapping an oracle_id to its pre-computed feature tensor
            anchor_size_range (tuple): A (min, max) tuple for the number of cards to randomly select for the anchor deck.
        """
        # Filter out any decks that are too small to sample from
        self.decklists = [deck for deck in decklists if len(deck) > 1]
        self.card_feature_map = card_feature_map
        self.all_card_ids = list(card_feature_map.keys()) #used for negative sampling
        self.anchor_size_range = anchor_size_range

    def __len__(self):
        return len(self.decklists)

    def __getitem__(self, idx):
        deck = self.decklists[idx]
        deck_set = set(deck)
        random.shuffle(deck) # Shuffle the deck in-place to ensure a different random split on every epoch

        max_possible_anchor_size = len(deck) - 1
        min_anchor = min(self.anchor_size_range[0], max_possible_anchor_size)
        max_anchor = min(self.anchor_size_range[1], max_possible_anchor_size)
        if min_anchor >= max_anchor:
            anchor_size = max_possible_anchor_size
        else:
            anchor_size = random.randint(min_anchor, max_anchor)

        anchor_deck_ids = deck[:anchor_size]
        holdout_ids = deck[anchor_size:]
        positive_card_id = random.choice(holdout_ids)

        # Find a Negative Card: a random card from the card pool not present in the original decklist.
        negative_card_id = None
        while negative_card_id is None or negative_card_id in deck_set:
            negative_card_id = random.choice(self.all_card_ids)

        positive_card_tensor = self.card_feature_map[positive_card_id]
        negative_card_tensor = self.card_feature_map[negative_card_id]
        
        anchor_deck_tensors = torch.stack([self.card_feature_map[id] for id in anchor_deck_ids])

        return anchor_deck_tensors, positive_card_tensor, negative_card_tensor


def triplet_collate_fn(batch):
    """
    A custom collate function to handle variable-length anchor decks in a batch:
    it pads the anchor decks to the same length and stacks the cards.
    """
    # Separate the components of the batch
    anchor_decks = [item[0] for item in batch]
    positive_cards = [item[1] for item in batch]
    negative_cards = [item[2] for item in batch]

    # Use pad_sequence to handle the variable-length anchor decks.
    # `batch_first=True` ensures the output shape is (Batch Size, Max_Seq_Len, Feature_Dim)
    padded_anchors = pad_sequence(anchor_decks, batch_first=True, padding_value=0.0)

    # Stack the positive and negative cards, which are already of a fixed size
    batched_positives = torch.stack(positive_cards)
    batched_negatives = torch.stack(negative_cards)

    return padded_anchors, batched_positives, batched_negatives