import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
from typing import List, Dict
import vector_database
import chromadb
import json
import pandas as pd
import seaborn as sns


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
        
        oracle_id = name_to_id_map.get(card_name.lower())

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


def search_results(this:str):
    data_dir = os.path.join(this, "data")
    repr_dict_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
    type_keyw_dict_path = os.path.join(data_dir, "type_and_keyw_dict.pt")
    card_dict_path = os.path.join(data_dir, "card_dict.pt")
    img_folder_path = os.path.join(data_dir, "images")
    
    models_dir = os.path.join(this, "models")
    mtg_llm_path = os.path.join(models_dir, "magic-distilbert-base-v1") # embedder for oracle texts
    role_class_path = os.path.join(models_dir, "card-role-classifier-final") # one-hot encoder for roles of cards

    db_path = os.path.join(this, "card_db")
    client = chromadb.PersistentClient(path=db_path)
    
    num_types = 422
    num_keyw = 627

    embedder_checkpoint_path = os.path.join(this, "models", "cpr_checkpoint_v1_all_200_3.pt")
    db_name = "mtg_cards_v1_all_200_3"

    embedder = vector_database.CardEmbedder(
        device="cpu",
        cpr_checkpoint_path=embedder_checkpoint_path,
        partial_map_path=repr_dict_path,
        cat_map_path=type_keyw_dict_path,
        llm_path=mtg_llm_path,
        role_class_path=role_class_path,
        num_types=num_types,
        num_keywords=num_keyw
    )

    retrieval_system = vector_database.CardRetriever(
        embedder=embedder,
        client=client,
        card_dict_path=card_dict_path
    )

    rielle_id = 11032857
    animar_id = 5096356
    grismold_id = 9151052

    user_prompt = ""
    #user_prompt = "sacrifice a creature"
    # user_prompt = "find some card draw"
    #user_prompt = "draw a card"
    #user_prompt = "deal damage to an opponent"
    #user_prompt = "destroy target creature"
    #user_prompt = "destroy all creatures"
    # user_prompt = "I need some creature removal"

    prompt_based_results = retrieval_system.recommend_cards(
        deck_id=animar_id,
        db_name=db_name,
        prompt=user_prompt
    )


    if prompt_based_results:
        show_card_results(
            card_names=prompt_based_results,
            card_dict_path=card_dict_path,
            image_folder=img_folder_path,
            title=f"Results for Prompt: '{user_prompt}', using collection '{db_name}'"
        )


def demo_statistics(this):
    feedback_path = os.path.join(this, "user_feedback.jsonl")
    if not os.path.exists(feedback_path):
        print(f"Error: Feedback file not found at '{feedback_path}'")
        return

    all_ratings_data = []
    with open(feedback_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                model_version = record.get('model_version')
                
                if not model_version or 'ratings' not in record:
                    continue
                
                # Create one entry for each individual card rating
                for rating_info in record['ratings']:
                    rating = rating_info.get('rating')
                    if rating is not None:
                        all_ratings_data.append({
                            'model_version': model_version,
                            'rating': rating
                        })
            except json.JSONDecodeError:
                print(f"Warning: Skipping a malformed line in the JSONL file.")
    
    if not all_ratings_data:
        print("No valid feedback records found in the file.")
        return

    df = pd.DataFrame(all_ratings_data)
    unique_models = df['model_version'].unique()
    n_models = len(unique_models)

    if n_models == 0:
        print("No models found in the feedback data.")
        return

    print(f"\nGenerating plots for {n_models} model(s)...")
    sns.set_theme(style="whitegrid", palette="viridis")

    n_cols = 4
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6), squeeze=False)
    axes = axes.flatten() # Simplifies later interaction

    for i, model_name in enumerate(sorted(unique_models)):
        ax = axes[i]
        model_df = df[df['model_version'] == model_name]
        sns.countplot(
            x='rating',
            data=model_df,
            ax=ax,
            order=[1, 2, 3, 4, 5] # Ensure all bars are shown, even if count is 0
        )

        total_ratings = len(model_df)
        avg_rating = model_df['rating'].mean()
        
        ax.set_title(f'"{model_name}"\n(Avg: {avg_rating:.2f}/5.00 from {total_ratings} ratings)', fontsize=8)
        ax.set_xlabel("User Rating", fontsize=10)
        ax.set_ylabel("Number of Ratings", fontsize=10)
        ax.set_ylim(0, max(10, model_df['rating'].value_counts().max() * 1.1)) # Dynamic y-axis limit

    for j in range(n_models, len(axes)):
        axes[j].set_visible(False) # Hide unused plots

    fig.suptitle("User Feedback Analysis by Model Version", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



if __name__ == '__main__':
    this = os.path.dirname(__file__)
    # search_results(this)
    demo_statistics(this)