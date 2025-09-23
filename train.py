import models
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import edh_scraper

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


def main_autoencoder_training(this):
    ## TRAINING IMAGE AUTOENCODER
    ae_dataset_path = os.path.join(this, "data", "img_dataset.pt")
    ae_checkpoint_path = os.path.join(this, "models", "ImgEncoder.pt")

    ## TRAINING WITH FIRST FUNCTION I CAME UP WITH
    train_and_save_convAE(ae_dataset_path, ae_checkpoint_path)

    ## TRAINING WITH GENERALIZED CLASS
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ae_model = models.HybridConvAutoencoder().to(DEVICE)
    ae_loss_fn = nn.MSELoss()
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=LEARNING_RATE)
    
    ae_full_dataset = torch.load(ae_dataset_path) # Img encoder wants to overfit to available images
    data_loader_ae = DataLoader(train_ds_ae, batch_size=8, shuffle=True)

    trainer_ae = Trainer(
        model=ae_model,
        optimizer=ae_optimizer,
        loss_fn=ae_loss_fn,
        train_loader=train_loader_ae,
        val_loader=None,
        checkpoint_path=ae_checkpoint_path,
        device=DEVICE
    )
    trainer_ae.train(NUM_EPOCHS, train_step_fn=autoencoder_step_fn)


# - - - - - - - - - - - - - - - - - Functions for contextual preference ranking pipeline - - - - - - - - - - - - - - - - - 


class Trainer:
    """
    General-purpose training class to handle model training, validation and checkpointing for different models
    """
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, checkpoint_path, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.scheduler = scheduler
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
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Resumed scheduler state.")
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
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
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

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR):
                 self.scheduler.step()

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

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): # for schedulers that need a metric to step
                    self.scheduler.step(val_loss)
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR): # all other schedulers
                     self.scheduler.step()

            if val_loss is not None:
                print(f"Epoch Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                # Save the model only if validation loss improves
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
            else:
                # If no validation, save after every epoch
                print(f"Epoch Summary: Train Loss: {train_loss:.6f}")
                self._save_checkpoint(epoch, train_loss)
        print("\n--- Training Finished ---")


def cpr_step_fn_triplet(model, batch, loss_fn, device, temperature=0.5):
    """
    Distance-weighted negative sampling step function for triplet loss.
    This function integrates the FeatureEncoder to create complete card representations before feeding them to the main encoders.
    """
    anchors, positives, anchor_types, anchor_keyw, pos_types, pos_keyw = batch
    anchor_decks = anchors.to(device)
    positive_cards = positives.to(device)
    anchor_types = anchor_types.to(device) # batch of sequences
    anchor_keyw = anchor_keyw.to(device)
    pos_types = pos_types.to(device) # batch of single items
    pos_keyw = pos_keyw.to(device)

    anchor_cat_emb = model.feature_encoder(anchor_types, anchor_keyw)
    pos_cat_emb = model.feature_encoder(pos_types, pos_keyw)

    # (B, Seq_Len, Num_Features) + (B, Seq_Len, Cat_Emb_Dim) -> (B, Seq_Len, Total_Dim)
    anchor_full = torch.cat((anchor_decks, anchor_cat_emb), dim=2)
    # (B, Num_Features) + (B, Cat_Emb_Dim) -> (B, Total_Dim)
    positive_full = torch.cat((positive_cards, pos_cat_emb), dim=1)
    
    anchor_emb = model.deck_embedding(anchor_full)
    pos_emb = model.card_embedding(positive_full)

    anchor_emb = F.normalize(anchor_emb, dim=1)
    pos_emb = F.normalize(pos_emb, dim=1)

    B = anchor_full.size(0)
    sims = torch.matmul(anchor_emb, pos_emb.T) / temperature
    mask = torch.eye(B, dtype=torch.bool, device=device)
    sims.masked_fill_(mask, float('-inf'))       # ignore self-similarity

    # Sample a negative based on distance-weighted probability
    weights = F.softmax(sims, dim=1)
    neg_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
    neg_emb = pos_emb[neg_indices]

    loss = loss_fn(anchor_emb, pos_emb, neg_emb)
    return loss


def cpr_step_fn_infonce(model, batch, loss_fn, device, temperature=0.5):
    """
    InfoNCE loss step function using dot product for similarity without vector normalization.
    This function integrates the FeatureEncoder to create complete card representations before feeding them to the main encoders.
    It treats other positive cards in the batch as negative samples for contrastive learning.
    """
    anchors, positives, anchor_types, anchor_keyw, pos_types, pos_keyw = batch
    anchors, positives = anchors.to(device), positives.to(device)
    anchor_types, anchor_keyw = anchor_types.to(device), anchor_keyw.to(device)
    pos_types, pos_keyw = pos_types.to(device), pos_keyw.to(device)

    # Encode categorical features
    anchor_cat_emb = model.feature_encoder(anchor_types, anchor_keyw)   # (B, Seq_Len, cat_dim)
    pos_cat_emb = model.feature_encoder(pos_types, pos_keyw)         # (B, cat_dim)

    # Concatenate partial features + categorical features
    anchor_full = torch.cat((anchors, anchor_cat_emb), dim=2)  # (B, Seq_Len, dim)
    pos_full = torch.cat((positives, pos_cat_emb), dim=1)   # (B, dim)

    # Get embeddings
    anchor_emb = model.deck_embedding(anchor_full)  # (B, D)
    pos_emb    = model.card_embedding(pos_full)     # (B, D)

    # Normalize for cosine similarity
    anchor_emb = F.normalize(anchor_emb, dim=1)
    pos_emb = F.normalize(pos_emb, dim=1)

    # Compute similarity matrix (B x B)
    logits = torch.matmul(anchor_emb, pos_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=device)

    loss = loss_fn(logits, labels)
    return loss


def main_cpr_training(
    cpr_dataset_path,
    cpr_checkpoint_path,
    loss_fn,
    step_fn,
    NUM_EPOCHS = 20,
    LEARNING_RATE = 0.00002,
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    BATCH_SIZE = 128,
    NUM_TYPES = 420,
    NUM_KEYW = 627):
    ### TRAINING CPR PIPELINE WITH GENERALIZED CLASS

    feature_encoder = models.FeatureEncoder(num_types=NUM_TYPES, type_emb_dim=64, num_keyw=NUM_KEYW, keyw_emb_dim=64)
    cpr_model = models.PipelineCPR(feature_encoder, out_dim=512).to(DEVICE)
    cpr_optimizer = optim.AdamW(cpr_model.parameters(), lr=LEARNING_RATE)

    cpr_full_dataset = torch.load(cpr_dataset_path, weights_only = False)
    train_ds_cpr, val_ds_cpr = torch.utils.data.random_split(cpr_full_dataset, [0.8, 0.2])
    train_loader_cpr = DataLoader(train_ds_cpr, batch_size=BATCH_SIZE, drop_last=False, collate_fn=models.triplet_collate_fn, shuffle=True)
    val_loader_cpr = DataLoader(val_ds_cpr, batch_size=BATCH_SIZE, drop_last=False, collate_fn=models.triplet_collate_fn, shuffle=False)

    num_training_steps = NUM_EPOCHS * len(train_loader_cpr)
    num_warmup_steps = int(0.05 * num_training_steps)
    cpr_scheduler = get_linear_schedule_with_warmup(
        optimizer=cpr_optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    trainer_cpr = Trainer(
        model=cpr_model,
        optimizer=cpr_optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader_cpr,
        val_loader=val_loader_cpr,
        checkpoint_path=cpr_checkpoint_path,
        device=DEVICE,
        scheduler=cpr_scheduler
    )

    trainer_cpr.train(NUM_EPOCHS, step_fn=step_fn)


# - - - - - - - - - - - - - - - - - Auxiliary functions, post training - - - - - - - - - - - - - - - - - 


def generate_and_save_emb_dict(card_feature_map_path, cat_feature_map_path, cpr_checkpoint_path, num_types, num_keyw, batch_size, out_path):
    """
    Generates and saves a dictionary mapping oracle_ids to their respective embedding
    """
    if not os.path.exists(card_feature_map_path):
        print(f"Card feature map not found at '{card_feature_map_path}'"); return
    card_feature_map = torch.load(card_feature_map_path)

    if not os.path.exists(cat_feature_map_path):
        print(f"Card feature map not found at '{cat_feature_map_path}'"); return
    cat_feature_map = torch.load(cat_feature_map_path) # made of items like  oracle_id: {"types": .., "keywords": ..}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    card_encoder = load_card_encoder(cpr_checkpoint_path, num_types, num_keyw, device)


    print("Preparing data for batching...")
    oids_list = []
    partial_tensors_list = []
    types_list = []
    keywords_list = []
    for oid, partial_repr in card_feature_map.items():
        oids_list.append(oid)
        partial_tensors_list.append(partial_repr)
        cat_data = cat_feature_map[oid]
        types_list.append(cat_data["types"])
        keywords_list.append(cat_data["keywords"])

    cards_embeddings = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(oids_list), batch_size), desc="Creating embeddings for all cards"):
            batch_oids = oids_list[i:i+batch_size]
            batch_partials = partial_tensors_list[i:i+batch_size]
            batch_types = types_list[i:i+batch_size]
            batch_keywords = keywords_list[i:i+batch_size]

            partials_tensor = torch.stack(batch_partials).to(device)
            types_tensor = torch.stack(batch_types).to(device)
            keywords_tensor = torch.stack(batch_keywords).to(device)
            
            cat_embeddings = card_encoder.feature_encoder(types_tensor, keywords_tensor)
            full_card_repr = torch.cat([partials_tensor, cat_embeddings], dim=1)
            
            batch_embeddings = card_encoder.card_embedding(full_card_repr)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1) # normalize to improve swearch results
            for oid, emb in zip(batch_oids, batch_embeddings):
                cards_embeddings[oid] = emb.cpu()
    
    torch.save(cards_embeddings, out_path)
    print(f"Successfully created card embedding mapping in {out_path}")


def load_card_encoder(path, num_types, num_keywords, device):
    """
    Initializes the PipelineCPR model with its required parameters
    before loading the saved state dictionary.
    """
    feature_encoder = models.FeatureEncoder(
        num_types=num_types,
        type_emb_dim=64,
        num_keyw=num_keywords,
        keyw_emb_dim=64
    )
    model = models.PipelineCPR(
        feature_encoder=feature_encoder,
        partial_card_dim=797, 
        card_hidden_dim=1024,
        embed_dim=512,
        out_dim=512
    ).to(device)

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model




if __name__ == "__main__":
    
    this = os.path.dirname(__file__)
    data_dir = os.path.join(this, "data")
    models_dir = os.path.join(this, "models")
    # main_autoencoder_training(this)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    # data_dir = os.path.join(this, "data")
    # # decks_path_all = os.path.join(data_dir, "edh_decks_all.jsonl")
    # # dataset_path_all = os.path.join(data_dir, "cpr_dataset_v1_all.pt")
    # decks_path_div = os.path.join(data_dir, "edh_decks_div.jsonl")
    # dataset_path_div = os.path.join(data_dir, "cpr_dataset_v1_div.pt")
    # card_feat_map_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
    # cat_feat_map_path = os.path.join(data_dir, "type_and_keyw_dict.pt")

    # edh_scraper.create_and_save_CPRdataset(decks_path_div, dataset_path_div, card_feat_map_path, cat_feat_map_path)

    # epochs_list = [20,200]
    # cpr_dataset_path = os.path.join(data_dir, "cpr_dataset_v1_div.pt")
    # loss_fn = nn.TripletMarginLoss(margin=0.3)
    # step_fn = cpr_step_fn_triplet
    # for epochs in epochs_list:
    #     cpr_checkpoint_path = os.path.join(this, "models", "cpr_checkpoint_v1_div_"+str(epochs)+"_3.pt")
    #     main_cpr_training(cpr_dataset_path, cpr_checkpoint_path, loss_fn, step_fn, epochs)

    # loss_fn = nn.CrossEntropyLoss()
    # step_fn = cpr_step_fn_infonce
    # for epochs in epochs_list:
    #     cpr_checkpoint_path = os.path.join(this, "models", "cpr_checkpoint_v1_div_"+str(epochs)+"_nce.pt")
    #     main_cpr_training(cpr_dataset_path, cpr_checkpoint_path, loss_fn, step_fn, epochs)

    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Removed some cards -> -2 types of cards
    # Retraining only the top 3 models from previous testing

    div_path = os.path.join(data_dir, "cpr_dataset_v1_div_s2.pt")
    all_path = os.path.join(data_dir, "cpr_dataset_v1_all_s2.pt")
    checkpoint_1 = os.path.join(models_dir, "cpr_checkpoint_v1_div_20_triplet_s2.pt")
    checkpoint_2 = os.path.join(models_dir, "cpr_checkpoint_v1_all_200_triplet_s2.pt")
    checkpoint_3 = os.path.join(models_dir, "cpr_checkpoint_v1_all_200_nce_s2.pt")
    
    main_cpr_training(div_path, checkpoint_1, nn.TripletMarginLoss(margin=0.3), cpr_step_fn_triplet, 20)
    main_cpr_training(all_path, checkpoint_2, nn.TripletMarginLoss(margin=0.3), cpr_step_fn_triplet, 200)
    main_cpr_training(all_path, checkpoint_3, nn.CrossEntropyLoss(), cpr_step_fn_infonce, 200)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    num_types = 420
    num_keyw = 627
    batch_size = 64

    repr_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
    type_keyw_path = os.path.join(data_dir, "type_and_keyw_dict.pt")

    # params = [
    # (os.path.join(data_dir, f"emb_dict_v1_{dataset}_{epochs}_{loss}.pt"),
    # os.path.join(this, "models", f"cpr_checkpoint_v1_{dataset}_{epochs}_{loss}.pt"))
    # for dataset in ["div", "all"] for epochs in ["20","200"] for loss in ["nce","3"]
    # ]
    
    params = [
        (os.path.join(data_dir, "emb_dict_v1_div_20_triplet_s2.pt"), checkpoint_1),
        (os.path.join(data_dir, "emb_dict_v1_all_200_triplet_s2.pt"), checkpoint_2),
        (os.path.join(data_dir, "emb_dict_v1_all_200_nce_s2.pt"), checkpoint_3)
    ]
    
    for emb_dict_path, cpr_checkpoint_path in params:
        generate_and_save_emb_dict(repr_path, type_keyw_path, cpr_checkpoint_path, num_types, num_keyw, batch_size, emb_dict_path)


