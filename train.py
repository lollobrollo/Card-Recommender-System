import models
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

def train_and_save_convAE(dataset_path, output_path, num_epochs=10, lr=1e-3, device=None):
    """
    Trains a Convolutional Autoencoder with checkpointing
    - Before training, checks if a checkpoint exists at 'output_path' to resume
    - At the end of each epoch, saves a new checkpoint with the model state, optimizer state, and epoch number.
    Args:
        dataset_path (str): Path to the dataset used for training
        output_path (str): Path where trained model will be saved
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        device (torch.device): Device to train on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = torch.load(dataset_path, weights_only=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # num_workers=4 (watch out for memory issues https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662)
    
    model = models.HybridConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0

    if os.path.exists(output_path):
        print(f"Checkpoint found at '{output_path}'. Attempting to resume training.")
        try:
            checkpoint = torch.load(output_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            last_loss = checkpoint.get('loss', 'N/A')
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
            print(f"Last saved epoch loss was: {last_loss}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
    else:
        print("No checkpoint found. Starting training from scratch.")

    model.train()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            imgs = batch.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")
    
    print("Saving checkpoint...")
    try:
        checkpoint = {
            'epoch': epoch + 1, # Save the *next* epoch number to start from
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }
        torch.save(checkpoint, output_path)
        print(f"Checkpoint saved successfully to '{output_path}'")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


if __name__ == "__main__":
    this = os.path.dirname(__file__)
    dataset_path = os.path.join(this, "data", "img_dataset.pt")
    output_path = os.path.join(this, "models", "ImgEncoder.pt")

    train_and_save_convAE(dataset_path, output_path)