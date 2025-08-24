from models import *
import torch
import torch.nn as nn
from torchvision import transforms
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import ijson
import re


def generate_and_save_dict():
    """
    Using data/clean_data.json, builds a case-insensitive map from card name to oracle_id
    Filters out cards without an associated image in data/images
    Saves the dictionary into data/card_dict.pt
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    json_path = os.path.join(data_dir, "clean_data.json")
    image_dir = os.path.join(data_dir, "images")
    output_path = os.path.join(data_dir, "card_dict.pt")
    
    # Get a set of all available image IDs
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Create a set of oracle_ids from the image filenames
    try:
        available_image_ids = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}
        print(f"Found {len(available_image_ids)} images in the '{os.path.basename(image_dir)}' directory.")
    except Exception as e:
        print(f"Error reading image directory {image_dir}: {e}")
        return

    # Iterate through the JSON and build the dictionary
    card_dict = {}
    print(f"Reading {os.path.basename(json_path)} to build dictionary...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for card in tqdm(ijson.items(f, 'item'), desc="Processing cards"):
                oracle_id = card.get('oracle_id')
                # Only add the card if its oracle_id has a corresponding image
                if oracle_id and oracle_id in available_image_ids:
                    name = card.get('name').lower()
                    card_dict[name] = oracle_id
    except FileNotFoundError:
        print(f"Error: Clean data file not found at {json_path}")
        return

    # Save the final dictionary
    if not card_dict:
        print("\nWarning: The final dictionary is empty. No matching images were found for cards in the JSON.")
        return
    print(f"\nGenerated dictionary with {len(card_dict)} cards.")
    try:
        torch.save(card_dict, output_path)
        print(f"Successfully saved dictionary to: {output_path}")
    except Exception as e:
        print(f"\nError saving dictionary to {output_path}: {e}")


def synchronize_images_and_data(json_path, image_dir):
    """
    Synchronizes the image directory with the clean data file by deleting
    any images that do not have a corresponding entry in the clean JSON data.
    (needed when filtering applied becomes more strict, to avoid downloading images from scratch)
    Args:
        json_path (str): The path to the final clean_data.json file
        image_dir (str): The path to the directory containing downloaded images
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_oracle_ids = {card.get('oracle_id') for card in ijson.items(f, 'item') if card.get('oracle_id')}
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}. Aborting sync.")
        return
    print(f"Found {len(json_oracle_ids)} unique cards in the clean data file.")

    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}. Nothing to sync.")
        return
        
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir)}
    image_oracle_ids = set(image_files.keys())
    print(f"Found {len(image_oracle_ids)} images in the '{os.path.basename(image_dir)}' directory.")

    ids_to_delete = image_oracle_ids - json_oracle_ids # Set difference operation
    if not ids_to_delete:
        print("\nImage directory is already in sync with the clean data. No files to delete.")
        return

    print(f"\nFound {len(ids_to_delete)} stale images to delete...")
    deleted_count = 0
    for oracle_id in tqdm(ids_to_delete, desc="Deleting stale images"):
        filename_to_delete = image_files[oracle_id]
        file_path_to_delete = os.path.join(image_dir, filename_to_delete)
        try:
            os.remove(file_path_to_delete)
            deleted_count += 1
        except OSError as e:
            tqdm.write(f"Error deleting file {file_path_to_delete}: {e}")
    print(f"\nSync complete. Deleted {deleted_count} stale images.")


def save_dataset_to_pt(img_dir, output_file):
    """
    Helper function that saves to file the dataset used to train the convolutional autoencoder.
    Args:
        img_dir (str): Path to folder containing images
        output_file (str): Path to output .pt file
    """
    dataset = CardImageDataset(img_dir)
    torch.save(dataset, output_file)


def load_img_encoder(checkpoint_path, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads the weithts of the trained autoencoder and returns an object containing the encoder for inference
    Args:
        checkpoint_path (str): Path to the checkpoint containing weights of the trained encoder
        device (str): Device to use the model on
    """
    encoder = HybridConvAutoencoder()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    encoder.load_state_dict(model_state_dict)
    encoder.to(device)
    encoder.eval()
    return encoder


def show_reconstructions(checkpoint_path, img_dir, device='cuda' if torch.cuda.is_available() else 'cpu', num_images=5):
    """
    Shows original card images and reconstrucitons side by side for some cards
    """
    
    model = load_img_encoder(checkpoint_path, device=device)
    
    transform = transforms.Compose([
        transforms.Resize((936, 672)),
        transforms.ToTensor(),
    ])

    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
    image_files = image_files[:num_images]

    original_images = []
    reconstructed_images = []
    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 936, 672)

            output = model(input_tensor)

            original_images.append(input_tensor.squeeze(0).cpu())
            reconstructed_images.append(output.squeeze(0).cpu())

    plt.figure(figsize=(12, 4 * num_images))
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2*i + 1)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(original_images[i].permute(1, 2, 0))

        # Reconstructed image
        plt.subplot(num_images, 2, 2*i + 2)
        plt.title("Reconstruction")
        plt.axis('off')
        plt.imshow(reconstructed_images[i].permute(1, 2, 0))

    plt.tight_layout()
    plt.show()


def get_all_card_types_and_keywords(json_path):
    """
    Scans the dataset and extracts all unique types, supertypes, and subtypes from the 'type_line' of each card.
    Also extracts all keywords present in cards.
    Args:
        json_path (str): The path to the clean_data.json file.
    Returns two Python ordered lists containing:
        all unique type strings found in the dataset
        all unique keywords found in the dataset
    """
    print(f"Extracting all unique types an keywords from: {os.path.basename(json_path)}")
    all_types = set()
    all_keywords = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for card in ijson.items(f, 'item'):
                    all_types.update(extract_card_types(card))
                    all_keywords.update(extract_card_keywords(card))

    except FileNotFoundError:
        print(f"Error: Data file not found at {json_path}")
        return set()
    print(f"Extraction complete.\nFound {len(all_types)} unique types, supertypes, and subtypes.\nFound {len(all_keywords)} unique keywords.")

    return sorted(list(all_types)), sorted(list(all_keywords)) 


def extract_card_keywords(card):
    """
    Helper function that extract and returns all keywords of the card passed as argument
    """
    card_keywords = card.get("keywords")
    if not card_keywords:
        return set()
    return set(card_keywords)


def extract_card_types(card):
    """
    Helper function that extract and returns all types of the card passed as argument
    """
    card_types_set = set()
    type_line = card.get("type_line")
    if not type_line:
        return set()
    if "—" in type_line:
        parts = type_line.split('—')
        main_types_part = parts[0]
        sub_types_part = parts[1]
    else:
        main_types_part = type_line
        sub_types_part = "" # Card has no subtypes (e.g., "Instant")

    main_types = main_types_part.strip().split()
    card_types_set.update(main_types)

    if sub_types_part:
        sub_types = sub_types_part.strip().split()
        card_types_set.update(sub_types)

    return card_types_set


def preprocess_oracle_text(text, remove_reminder=True, mana_as_words=True, mask_name=True):
    """
    Preprocess Oracle text for sentence encoding
    Args:
        text (str): The raw oracle text from Scryfall
        remove_reminder (bool): If True, strips reminder text in parentheses (keywords encoded separately, could be redundant)
        mana_as_words (bool): If True, replaces mana symbols like {G} with words ("green mana")
        mask_name (bool): If True, replace most istances of card self reference with a generic placeholder
    Returns:
        str: The cleaned Oracle text.
    """
    if not text:
        return ""

    # 1. Normalize newlines to separate abilities with a period + space
    text = text.strip()
    text = re.sub(r"\s*\n\s*", ". ", text)

    # 2. Remove reminder text (text inside parentheses)
    if remove_reminder:
        text = re.sub(r"\([^)]*\)", "", text)

    # 3. Replace mana symbols with words
    if mana_as_words:
        mana_map = {
            "{W}": "white mana",
            "{U}": "blue mana",
            "{B}": "black mana",
            "{R}": "red mana",
            "{G}": "green mana",
            "{C}": "colorless mana",
            "{T}": "tap",
            "{Q}": "untap"
        }
        # Add generic mana costs like {2}, {3}, etc.
        mana_map.update({f"{{{i}}}": f"{i} generic mana" for i in range(0, 21)})
        for symbol, replacement in mana_map.items():
            text = text.replace(symbol, replacement)

    import re

    # 4. Replace card self references with placeholder
    if mask_name:
        # Escape card name for regex, match whole word case-insensitive
        card_name_pattern = re.escape(card_name)
        # \b for word boundaries so partial matches don't occur
        card_name_regex = re.compile(rf'\b{card_name_pattern}\b', flags=re.IGNORECASE)
        # Phrases that commonly refer to the card itself
        self_ref_phrases = [
            r'this card',
            r'this creature',
            r'this artifact',
            r'this enchantment',
            r'this planeswalker',
            r'this land',
            r'this permanent',
        ]
        # Compile regex for self reference phrases (word boundaries + case insensitive)
        self_ref_regex = re.compile(r'\b(' + '|'.join(self_ref_phrases) + r')\b', flags=re.IGNORECASE)
        
        text = card_name_regex.sub('[CARD]', text)
        text = self_ref_regex.sub('[CARD]', text)

    # 5. Normalize spaces
    text = re.sub(r"\s+", " ", text)

    # 6. Trim leftover punctuation spacing
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = text.strip()

    return text


def create_and_save_dataset(decks_path: str, output_path: str, card_feature_map_path: str):
    """
    Creastes and saves a dataset for the training of the PipelineCPR model
    Extracts required informations from the decks scraped from the web (archidekt)
    """
    max_deck_size = 0; min_deck_size = 101
    decklists = []
    with open(decks_path, "r", encoding="utf-8") as decks:
        for deck in decks:
            cards = deck.get("mainboard", [])
            cards.extenddeck.get("commander_ids", [])
            decklists.append(cards)
            max_deck_size = max(max_deck_size, len(cards))
            min_deck_size = min(min_deck_size, len(cards))

    anchor_size_range = (min_deck_size, max_deck_size)
    card_feature_map = torch.load(card_feature_map_path)

    dataset = TripletEDHDataset(decklists, card_feature_map, anchor_size_range)
    torch.save(dataset, output_path)


if __name__ == "__main__":
    this = os.path.dirname(__file__)
    img_dir = os.path.join(this, "data", "images")
    dataset_path = os.path.join(this, "data", "img_dataset.pt")
    checkpoint_path = os.path.join(this, "models", "ImgEncoder.pt")
    generate_and_save_dict()
    # save_dataset_to_pt(img_dir, dataset_path)
    # show_reconstructions(checkpoint_path, img_dir)