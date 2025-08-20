import models
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm


# - - - - - - - - - - - - - - - - - Training cycle for card image representation - - - - - - - - - - - - - - - - - 

# Can be substituted with the general version below
def train_and_save_convAE(dataset_path, output_path, num_epochs=10, lr=1e-3, device=None):
    """
    Trains a Convolutional Autoencoder with checkpointing
    Args:
        dataset_path (str): Path to the dataset used for training (no train_test_split, as this wants to overfit)
        output_path (str): Path where trained model will be saved
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        device (str): Device to train on
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
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.6f}")
    
        print("Saving checkpoint...")
        try:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, output_path)
            print(f"Checkpoint saved successfully to '{output_path}'")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


# Step function used in the general train class below
def autoencoder_step_fn(model, batch, loss_fn, device):
    imgs = batch.to(device)
    reconstructed_imgs = model(imgs)
    loss = loss_fn(reconstructed_imgs, imgs)
    return loss


# - - - - - - - - - - - - - - - - - Models for contextual preference ranking pipeline - - - - - - - - - - - - - - - - - 


class Trainer:
    """
    General-purpose training class to handle model training, validation and checkpointing for different models
    """
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, checkpoint_path, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.start_epoch = 0
        self.best_val_loss = float('inf')

        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print(f"Checkpoint found at '{self.checkpoint_path}'. Resuming training.")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch']
                self.best_val_loss = checkpoint.get('loss', float('inf'))
                print(f"Resumed from epoch {self.start_epoch}. Best val loss: {self.best_val_loss:.6f}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
        else:
            print("No checkpoint found. Starting training from scratch.")

    def _save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved to '{self.checkpoint_path}'")

    def _train_one_epoch(self, step_fn):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            loss = step_fn(self.model, batch, self.loss_fn, self.device)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def _validate_one_epoch(self, step_fn):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                loss = step_fn(self.model, batch, self.loss_fn, self.device)
                running_loss += loss.item()
        return running_loss / len(self.val_loader)

    def train(self, num_epochs, step_fn):
        for epoch in range(self.start_epoch, num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

            train_loss = self._train_one_epoch(step_fn)
            val_loss = self._validate_one_epoch(step_fn)

            if val_loss is not None:
                print(f"Epoch Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                # Save the model only if validation loss improves
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"New best validation loss! Saving model...")
                    self._save_checkpoint(epoch, val_loss)
            else:
                # If no validation, save after every epoch
                print(f"Epoch Summary: Train Loss: {train_loss:.6f}")
                self._save_checkpoint(epoch, train_loss)
        print("\n--- Training Finished ---")


def cpr_step_fn(model, batch, loss_fn, device):
    anchor_deck, positive_card, negative_card = batch
    anchor_deck = anchor_deck.to(device)
    positive_card = positive_card.to(device)
    negative_card = negative_card.to(device)

    deck_emb, pos_emb, neg_emb = model(anchor_deck, positive_card, negative_card)
    loss = loss_fn(deck_emb, pos_emb, neg_emb)
    return loss




if __name__ == "__main__":
    this = os.path.dirname(__file__)

    ### TRAINING IMAGE AUTOENCODER
    # ae_dataset_path = os.path.join(this, "data", "img_dataset.pt")
    # ae_checkpoint_path = os.path.join(this, "models", "ImgEncoder.pt")

    ### TRAINING WITH FIRST FUNCTION I CAME UP WITH
    # train_and_save_convAE(ae_dataset_path, ae_checkpoint_path)

    ### TRAINING WITH GENERALIZED CLASS
    # NUM_EPOCHS = 10
    # LEARNING_RATE = 1e-3
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ae_model = models.HybridConvAutoencoder().to(DEVICE)
    # ae_loss_fn = nn.MSELoss()
    # ae_optimizer = optim.Adam(ae_model.parameters(), lr=LEARNING_RATE)
    
    # ae_full_dataset = torch.load(ae_dataset_path) # Img encoder wants to overfit to available images
    # data_loader_ae = DataLoader(train_ds_ae, batch_size=8, shuffle=True)

    # trainer_ae = Trainer(
    #     model=ae_model,
    #     optimizer=ae_optimizer,
    #     loss_fn=ae_loss_fn,
    #     train_loader=train_loader_ae,
    #     val_loader=None,
    #     checkpoint_path=ae_checkpoint_path,
    #     device=DEVICE
    # )
    # trainer_ae.train(NUM_EPOCHS, train_step_fn=autoencoder_step_fn)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ### TRAINING CPR PIPELINE WITH GENERALIZED CLASS
    cpr_dataset_path = os.path.join(this, "data", "cpr_dataset.pt")
    cpr_checkpoint_path = os.path.join(this, "models", "cpr_checkpoint.pt")

    cpr_model = models.PipelineCPR(card_vector_dim=1446, embedding_dim=512).to(DEVICE)
    cpr_loss_fn = models.TripletLoss(margin=1.0) # Assuming TripletLoss is in your models file
    # cpr_optimizer = optim.Adam(cpr_model.parameters(), lr=LEARNING_RATE)

    # 3. Create DataLoaders
    # cpr_full_dataset = torch.load(cpr_dataset_path)
    # train_ds_cpr, val_ds_cpr = torch.utils.data.random_split(cpr_full_dataset, [train_size, val_size])
    # train_loader_cpr = DataLoader(train_ds_cpr, batch_size=32, shuffle=True)
    # val_loader_cpr = DataLoader(val_ds_cpr, batch_size=32, shuffle=False)

    # # 4. Create and run the Trainer
    # trainer_cpr = Trainer(
    #     model=cpr_model,
    #     optimizer=cpr_optimizer,
    #     loss_fn=cpr_loss_fn,
    #     train_loader=train_loader_cpr,
    #     val_loader=val_loader_cpr,
    #     checkpoint_path=cpr_checkpoint_path,
    #     device=DEVICE
    # )
    # trainer_cpr.train(NUM_EPOCHS, train_step_fn=cpr_step_fn)