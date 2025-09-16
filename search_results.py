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
import re


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


def demo_statistics_v1(this):
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

                for rating_info in record['ratings']:
                    rating = rating_info.get('rating')
                    if rating is not None:
                        all_ratings_data.append({
                            'model_version': model_version,
                            'rating': rating
                        })
            except json.JSONDecodeError:
                print("Warning: Skipping a malformed line in the JSONL file.")
    
    if not all_ratings_data:
        print("No valid feedback records found in the file.")
        return

    df = pd.DataFrame.from_records(all_ratings_data)

    pattern = re.compile(r"(Diversified|Complete) Dataset \((\d+) Epochs, (Triplet|InfoNCE)\)")
    df[['dataset', 'epochs', 'loss']] = df['model_version'].str.extract(pattern)
    df = df.dropna(subset=["dataset", "epochs", "loss"])
    df['epochs'] = pd.to_numeric(df['epochs'])

    # Compute stats for later titles
    stats = df.groupby(['dataset', 'loss', 'epochs'])['rating'].agg(['mean', 'count']).reset_index()

    # --- Single combined grid ---
    g = sns.catplot(
        data=df,
        x="rating",
        col="epochs",
        row="dataset",
        hue="loss",
        kind="count",
        palette="rocket",
        height=5,
        aspect=1.2,
        order=[1, 2, 3, 4, 5],
        legend=True,
        margin_titles=True
    )

    # Global title
    g.fig.suptitle("User Feedback Analysis (All Training Epochs)", fontsize=14, fontweight="bold")

    # Annotate each subplot
    for (row_val, col_val), ax in g.axes_dict.items():
        for epoch_count in sorted(df['epochs'].unique()):
            stat_row = stats[(stats["dataset"] == row_val) & (stats["loss"] == col_val) & (stats["epochs"] == epoch_count)]
            if not stat_row.empty:
                avg = stat_row["mean"].iloc[0]
                count = stat_row["count"].iloc[0]
                ax.set_title(f"{row_val} | {col_val} ({epoch_count} epochs)\n(Avg: {avg:.2f}/5, N={count})", fontsize=9)

    g.set_axis_labels("User Rating", "Number of Ratings")
    g.figure.subplots_adjust(top=0.85)

    plt.show()


def demo_statistics(this):
    feedback_path = os.path.join(this, "user_feedback.jsonl")
    if not os.path.exists(feedback_path):
        print(f"Error: Feedback file not found at '{feedback_path}'")
        return

    # ... (Data loading and DataFrame creation is the same)
    all_ratings_data = []
    with open(feedback_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                model_version = record.get('model_version')
                if not model_version or 'ratings' not in record: continue
                for rating_info in record['ratings']:
                    all_ratings_data.append({
                        'model_version': model_version,
                        'rating': rating_info.get('rating')
                    })
            except json.JSONDecodeError:
                continue
    if not all_ratings_data:
        print("No valid feedback records found.")
        return
    df = pd.DataFrame.from_records(all_ratings_data)
    pattern = re.compile(r"(Diversified|Complete) Dataset \((\d+) Epochs, (Triplet|InfoNCE)\)")
    df[['dataset', 'epochs', 'loss']] = df['model_version'].str.extract(pattern)
    df = df.dropna(subset=["dataset", "epochs", "loss"])
    df['epochs'] = pd.to_numeric(df['epochs'])
    # ... (End of unchanged section)

    # 1. Compute stats for all combinations
    stats = df.groupby(['dataset', 'epochs', 'loss'])['rating'].agg(['mean', 'count']).reset_index()

    # 2. Use catplot to create the faceted grid
    g = sns.catplot(
        data=df,
        x="rating",
        col="loss",
        row="dataset",
        hue="epochs",
        kind="count",
        palette="tab10",
        height=5,
        aspect=1.2,
        order=[1, 2, 3, 4, 5],
        legend_out=True
    )

    # 3. Add a clear, global title
    g.fig.suptitle("User Feedback Analysis by Training Configuration", fontsize=16, fontweight="bold")
    g.fig.subplots_adjust(top=0.9)

    # --- 4. THE FIX IS HERE: Annotate each subplot using ax.text() ---
    for (row_val, col_val), ax in g.axes_dict.items():
        # Set the simple, clean title provided by Seaborn
        ax.set_title(f"{row_val} | {col_val}")
        
        # Get all stats for this specific subplot
        subplot_stats = stats[(stats['dataset'] == row_val) & (stats['loss'] == col_val)]
        
        # Build the multi-line string for our statistics box
        stats_lines = []
        for _, row in subplot_stats.iterrows():
            avg = row['mean']
            count = row['count']
            epochs = int(row['epochs'])
            stats_lines.append(f"{epochs} Epochs: Avg {avg:.2f}, N={count}")
        stats_string = "\n".join(stats_lines)

        # Place the text box in the top-right corner of the plot area.
        # `transform=ax.transAxes` uses coordinates relative to the plot axes (0,0 to 1,1).
        ax.text(0.95, 0.95, stats_string,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

        # Add text labels on top of each bar for clarity
        for p in ax.patches:
            height = p.get_height()
            if height > 0: # Only add labels to bars that exist
                ax.annotate(f'{int(height)}', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='center', 
                            xytext=(0, 5), 
                            textcoords='offset points')
    # --- END OF FIX ---
    
    # 5. Finalize and show the plot
    g.set_axis_labels("User Rating", "Number of Ratings")
    plt.show()


if __name__ == '__main__':
    this = os.path.dirname(__file__)
    # search_results(this)
    demo_statistics(this)