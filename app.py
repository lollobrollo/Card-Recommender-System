"""
    Author : Lorenzo Bortolussi
    Year : 2024/2025
    This code is part of the implementation of the project developed for my Thesis in Artificial Intelligence and Data Analytics.
"""

import gradio as gr
import torch
import os
import re
from PIL import Image
import json
from datetime import datetime
import chromadb
import vector_database
import utils


this_dir = os.path.dirname(__file__)
data_dir = os.path.join(this_dir, "data")
models_dir = os.path.join(this_dir, "models")
db_path = os.path.join(this_dir, "card_db")
img_folder_path = os.path.join(data_dir, "images")
feedback_file = os.path.join(this_dir, "misc", "user_feedback_s2.jsonl") # file that will hold user feedback 

# model_versions = {
#     f"{ds[0]} Dataset ({epochs} Epochs, {loss[0]})": {
#         "checkpoint": os.path.join(models_dir, f"cpr_checkpoint_v1_{ds[1]}_{epochs}_{loss[1]}.pt"),
#         "db_name": f"mtg_cards_v1_{ds[1]}_{epochs}_{loss[1]}"
#     }
#     for ds in [("Diversified", "div"), ("Complete", "all")]
#     for epochs in ["20", "200"]
#     for loss in [("Triplet", "3"), ("InfoNCE", "nce")]
# }

# Models that performed best in the first testing
model_versions = {
    "Diversified Dataset (20 Epochs, Triplet) s2": {
        "checkpoint": os.path.join(models_dir, "cpr_checkpoint_v1_div_20_triplet_s2.pt"),
        "db_name": "mtg_cards_v1_div_20_triplet_s2"
    },
    "Complete Dataset (200 Epochs, Triplet) s2": {
        "checkpoint": os.path.join(models_dir, "cpr_checkpoint_v1_all_200_triplet_s2.pt"),
        "db_name": "mtg_cards_v1_all_200_triplet_s2"
    },
    "Complete Dataset (200 Epochs, InfoNCE) s2": {
        "checkpoint": os.path.join(models_dir, "cpr_checkpoint_v1_all_200_nce_s2.pt"),
        "db_name": "mtg_cards_v1_all_200_nce_s2"
    }
}

# Add other three for personal use
# model_versions.update({
#     "Diversified Dataset (20 Epochs, Triplet)": {
#         "checkpoint": os.path.join(models_dir, "cpr_checkpoint_v1_div_20_3.pt"),
#         "db_name": "mtg_cards_v1_div_20_3"
#     },
#     "Complete Dataset (200 Epochs, Triplet)": {
#         "checkpoint": os.path.join(models_dir, "cpr_checkpoint_v1_all_200_3.pt"),
#         "db_name": "mtg_cards_v1_all_200_3"
#     },
#     "Complete Dataset (200 Epochs, InfoNCE)": {
#         "checkpoint": os.path.join(models_dir, "cpr_checkpoint_v1_all_200_nce.pt"),
#         "db_name": "mtg_cards_v1_all_200_nce"
#     }
# })



print("Loading shared resources...")
repr_dict_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
type_keyw_dict_path = os.path.join(data_dir, "type_and_keyw_dict.pt")
type_keyw_dict_s2_path = os.path.join(data_dir, "type_and_keyw_dict_s2.pt")
card_dict_path = os.path.join(data_dir, "card_dict.pt")
mtg_llm_path = os.path.join(models_dir, "magic-distilbert-base-v1")
role_class_path = os.path.join(models_dir, "card-role-classifier-final")
client = chromadb.PersistentClient(path=db_path)
name_to_id_map = torch.load(card_dict_path, weights_only=False)
id_to_name_map = {v: k for k, v in name_to_id_map.items()} # used for logging

num_types = 420 # 420 in new versions (s2), old models still use 422
num_keyw = 627

retriever_cache = {}

def get_retriever(model_version_name: str):
    if model_version_name in retriever_cache:
        print(f"Using cached retriever for '{model_version_name}'")
        return retriever_cache[model_version_name]

    print(f"Loading retriever for '{model_version_name}' for the first time...")
    config = model_versions[model_version_name]
    
    # Kinda patchwork, change num_types and type_and_keyw_dict based on model name
    if "s2" in model_version_name:
        num_types = 420
        embedder = vector_database.CardEmbedder(
            device="cuda" if torch.cuda.is_available() else "cpu",
            cpr_checkpoint_path=config["checkpoint"],
            partial_map_path=repr_dict_path,
            cat_map_path=type_keyw_dict_s2_path,
            llm_path=mtg_llm_path,
            role_class_path=role_class_path,
            num_types=420,
            num_keywords=num_keyw
        )
    else:
        embedder = vector_database.CardEmbedder(
            device="cuda" if torch.cuda.is_available() else "cpu",
            cpr_checkpoint_path=config["checkpoint"],
            partial_map_path=repr_dict_path,
            cat_map_path=type_keyw_dict_path,
            llm_path=mtg_llm_path,
            role_class_path=role_class_path,
            num_types=422,
            num_keywords=num_keyw
        )

    retriever = vector_database.CardRetriever(
        embedder=embedder,
        client=client,
        card_dict_path=card_dict_path
    )
    
    retriever_cache[model_version_name] = retriever
    print("Loading complete.")
    return retriever


# --- CORE FUNCTIONS FOR THE DEMO ---

def get_recommendations_and_show_feedback_ui(deck_url: str, prompt: str, model_choice: str):
    """
    This function returns updates for the gallery, the state, the feedback box visibility,
    and for the labels and visibility of each individual slider.
    """
    if not deck_url:
        raise gr.Error("Please provide a deck link.")

    match = re.search(r'/decks/(\d+)', deck_url)
    if not match:
        raise gr.Error("Could not find a valid deck ID in the URL.")
    deck_id = int(match.group(1))

    print(f"Processing Deck ID: {deck_id}, Model: {model_choice}, Prompt: '{prompt}'")
    
    retriever = get_retriever(model_choice)
    db_name = model_versions[model_choice]["db_name"]
    
    deck_oids, _, deck_colors = retriever._process_deck_for_search(deck_id, retriever.client.get_collection(name=db_name))
    color_identity_str = "".join(sorted(deck_colors)) if deck_colors else "C"

    recommended_names = retriever.recommend_cards(
        deck_id=deck_id,
        db_name=db_name,
        prompt=prompt if prompt else "",
        n=10
    )
    
    # --- DYNAMIC UI UPDATE LOGIC ---
    
    gallery_output = []
    card_ids_for_state = []
    slider_updates = []
    if not recommended_names:
        gr.Warning("No recommendations found for this query.")
        slider_updates = [gr.update(visible=False) for _ in range(10)]
        return [[]] + [None] + [gr.update(visible=False)] + slider_updates

    for name in recommended_names:
        oracle_id = name_to_id_map.get(name.lower())
        if oracle_id:
            img_path = os.path.join(img_folder_path, f"{oracle_id}.jpg")
            if os.path.exists(img_path):
                gallery_output.append((img_path, name))
                card_ids_for_state.append(oracle_id)
                # Update the slider to be visible and have the card name as its label
                slider_updates.append(gr.update(label=name, visible=True))

    while len(slider_updates) < 10:
        slider_updates.append(gr.update(visible=False))
            
    state_info = {
        "deck_id": deck_id, "prompt": prompt, "model_choice": model_choice,
        "color_identity": color_identity_str, "recommended_ids": card_ids_for_state
    }

    return [gallery_output] + [state_info] + [gr.update(visible=True)] + slider_updates


def save_feedback(state_info:dict, *ratings):
    if not state_info:
        gr.Warning("Cannot submit feedback without first getting recommendations.")
        return "Please generate recommendations before submitting feedback."

    feedback_record = {
        "timestamp": datetime.now().isoformat(),
        "model_version": state_info["model_choice"],
        "deck_id": state_info["deck_id"],
        "color_identity": state_info["color_identity"],
        "prompt": state_info["prompt"],
        "ratings": []
    }

    for i, card_id in enumerate(state_info["recommended_ids"]):
        feedback_record["ratings"].append({
            "oracle_id": card_id,
            "card_name": id_to_name_map.get(card_id, "Unknown"),
            "rating": int(ratings[i])
        })
        
    with open(feedback_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback_record) + '\n')

    gr.Info("Feedback submitted successfully. Thank you!")
    return "Feedback Submitted!"


# ---  GRADIO INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MTG Card Recommender System (with Feedback)")
    
    state = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=1):
            deck_url_input = gr.Textbox(label="Archidekt Deck URL")
            prompt_input = gr.Textbox(label="Optional Prompt")
            model_input = gr.Dropdown(
                label="Select Model Version",
                choices=list(model_versions.keys()),
                value=list(model_versions.keys())[0]
            )
            get_recs_button = gr.Button("Get Recommendations", variant="primary")
        
        with gr.Column(scale=2):
            gallery_output = gr.Gallery(
                label="Recommended Cards",
                columns=5, rows=2, object_fit="contain", height="auto"
            )

    with gr.Group(visible=False) as feedback_box:
        gr.Markdown("### Rate the Recommendations")
        gr.Markdown("Please rate each card on a scale of 1 (bad) to 5 (excellent).")
        
        sliders = []
        with gr.Row():
            for i in range(5):
                sliders.append(gr.Slider(minimum=1, maximum=5, step=1, label=f"Card {i+1}", value=3, elem_id=f"slider_{i}"))
        with gr.Row():
            for i in range(5):
                sliders.append(gr.Slider(minimum=1, maximum=5, step=1, label=f"Card {i+6}", value=3, elem_id=f"slider_{i+5}"))

        with gr.Row():
            submit_feedback_button = gr.Button("Submit Feedback", variant="primary")
            feedback_status = gr.Textbox(label="Status", interactive=False)

    all_outputs = [gallery_output, state, feedback_box] + sliders
    
    get_recs_button.click(
        fn=get_recommendations_and_show_feedback_ui,
        inputs=[deck_url_input, prompt_input, model_input],
        outputs=all_outputs
    )

    submit_feedback_button.click(
        fn=save_feedback,
        inputs=[state] + sliders,
        outputs=[feedback_status]
    )

if __name__ == "__main__":
    for model_version in list(model_versions.keys()):
        get_retriever(model_version)
    # get_retriever(list(model_versions.keys())[0])
    demo.launch(share=True)