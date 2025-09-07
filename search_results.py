import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
from typing import List, Dict
import vector_database
import chromadb


def normalize_card_name(name: str) -> str:
    """Normalizes a card name for consistent mapping."""
    name = name.lower()
    if ' // ' in name: name = name.split(' // ')[0]
    import re
    name = re.sub(r"[^\w\s]", '', name)
    return name.strip()


def show_card_results(card_names, card_dict_path, image_folder, title):
    """
    Displays the images of recommended cards in a grid.
    Args:
        card_names (List[str]): A list of card names from a search result.
        card_dict_path (Dict[str, str]): Path to a dictionary mapping normalized card names to oracle_ids.
        image_folder (str): The path to the directory where images are stored, named as {oracle_id}.jpg.
        title (str): The main title for the plot, highlighting the database used.
    """
    name_to_id_map = torch.load(card_dict_path, weights_only=False)

    if not card_names:
        print("No card names provided to display.")
        return

    num_cards = len(card_names)

    cols = min(num_cards, 5)
    rows = math.ceil(num_cards / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4.2))
    axes = axes.flatten() if num_cards > 1 else [axes]

    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, card_name in enumerate(card_names):
        ax = axes[i]
        
        norm_name = normalize_card_name(card_name)
        oracle_id = name_to_id_map.get(norm_name)

        if oracle_id:
            img_path = os.path.join(image_folder, f"{oracle_id}.jpg")
            
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image:\n{e}", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Oracle ID not found", ha='center', va='center')
        
        ax.set_title(card_name, fontsize=10)
        ax.axis('off')

    for j in range(num_cards, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    this = os.path.dirname(__file__)
    data_dir = os.path.join(this, "data")
    repr_dict_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
    type_keyw_dict_path = os.path.join(data_dir, "type_and_keyw_dict.pt")
    embedder_checkpoint_path = os.path.join(this, "models", "cpr_checkpoint_v1_all_200_3.pt")
    mtg_llm_path = os.path.join(this, "models", "magic-distilbert-base-v1")
    card_dict_path = os.path.join(data_dir, "card_dict.pt")
    img_folder_path = os.path.join(data_dir, "images")

    db_path = os.path.join(this, "card_db")
    client = chromadb.PersistentClient(path=db_path)
    db_name = "mtg_cards_v1_all_200_3"
    
    num_types = 422
    num_keyw = 627

    embedder = vector_database.CardEmbedder(
        device="cpu",
        cpr_checkpoint_path=embedder_checkpoint_path,
        partial_map_path=repr_dict_path,
        cat_map_path=type_keyw_dict_path,
        llm_path=mtg_llm_path,
        num_types=num_types,
        num_keywords=num_keyw
    )

    rielle_id = 11032857
    animar_id = 5096356
    grismold_id = 9151052

    user_prompt = "Find some efficient card draw spells"

    prompt_based_results = vector_database.recommend_cards_with_prompt(
        deck_id=rielle_id,
        prompt=user_prompt,
        n=10,
        embedder=embedder,
        client=client,
        db_name=db_name,
        alpha=0.7
    )

    if prompt_based_results:
        show_card_results(
            card_names=prompt_based_results,
            card_dict_path=card_dict_path,
            image_folder=img_folder_path,
            title=f"Results for Prompt: '{user_prompt}'"
        )