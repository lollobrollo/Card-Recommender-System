# chromadb docs 
# https://cookbook.chromadb.dev/core/filters/
# https://docs.trychroma.com/docs/overview/introduction

import chromadb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import models
from utils import preprocess_text
from train import load_card_encoder
import ijson
import os
import edh_scraper
import utils
from more_itertools import chunked
from itertools import combinations
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from rank_bm25 import BM25Okapi


# - - - - - - - - - - - - - Collection construction - - - - - - - - - - - - -

def build_and_save_chroma_db(card_to_embedding_path, cards_metadata_path, client, db_name):
    """
    Loads card features, generates embeddings using a trained model and saves them into a persistent ChromaDB database.
    """
    print("--- Building the Card Vector Database ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(card_to_embedding_path):
        print(f"Card to embedding mapping not found at '{card_to_embedding_path}'"); return
    card_emb_map = torch.load(card_to_embedding_path, weights_only = False)

    print(f"Initializing persistent ChromaDB.")
    card_collection = client.get_or_create_collection(name=db_name)
    
    card_metadata_map = build_metadata_map(cards_metadata_path, card_emb_map.keys())

    print("Preparing metadata for all cards...")
    all_ids = []
    all_embeddings = []
    all_metadatas = []

    with torch.no_grad():
        for oracle_id, embedding in card_emb_map.items():
            all_ids.append(oracle_id)
            all_embeddings.append(embedding.tolist())
            metadata = card_metadata_map.get(oracle_id, {})
            all_metadatas.append(metadata)
    
    chunk_size = 5000
    num_batches = len(all_ids) // chunk_size + bool(len(all_ids)%chunk_size)
    print(f"Adding {len(all_ids)} cards to the ChromaDB collection...")
    for batch_ids, batch_embs, batch_meta in tqdm(zip(
        chunked(all_ids, chunk_size),
        chunked(all_embeddings, chunk_size),
        chunked(all_metadatas, chunk_size)), total=num_batches
    ): # chunking since max batch size for collection.add() is 5461
        card_collection.add(
            ids=batch_ids,
            embeddings=batch_embs,
            metadatas=batch_meta
        )
    print(f"Vector database has been built and saved.")
    print(f"Total cards in collection: {card_collection.count()}")


def format_list_for_chroma(data: list) -> str:
    """
    Takes a list of strings and formats it into a single, delimited string safe for ChromaDB metadata
    Commas at the start and end make LIKE queries exact
    Example: ['Flying', 'Vigilance'] -> ",Flying,Vigilance,"
    Example: ['U', 'W'] -> ",U,W,"
    """
    if not data: return ","
    return f",{','.join(data)},"


def build_metadata_map(clean_data_path, all_oracle_ids):
    """
    Builds a dictionary mapping oracle_id to its clean, filter-ready metadata
    """
    print("Generating cards metadata map... ", end = "")
    card_metadata_map = {}
    with open(clean_data_path, 'r', encoding='utf-8') as f:
        mapped_ids = set(all_oracle_ids)

        for card in ijson.items(f, "item"):
            oracle_id = card.get("oracle_id")
            
            if oracle_id and oracle_id in mapped_ids:
                color_identity_list = card.get("color_identity")
                if not color_identity_list:
                    color_identity_list = ["C"]

                keywords_list = card.get("keywords", [])

                mana_cost_str = card.get("mana_cost", "")
                mana_details = utils.parse_mana_cost(mana_cost_str)
                mana_colors_list = mana_details.get('colors', [])

                color_identity_str = format_list_for_chroma(color_identity_list)
                keywords_str = format_list_for_chroma(keywords_list)
                mana_colors_str = format_list_for_chroma(mana_colors_list)

                card_metadata_map[oracle_id] = {
                    "name": card.get("name", "Unknown"),
                    "color_identity": color_identity_str, # e.g., ",U,W,"
                    "cmc": mana_details['cmc'],
                    "mana_colors": mana_colors_str, # e.g., ",U,W,"
                    "x_cost": mana_details['x_cost'],
                    "type_line": card.get("type_line", ""),
                    "keywords": keywords_str, # e.g., ",Flying,Vigilance,"
                    "oracle_text": card.get("oracle_text", "")
                }
    print("done")
    return card_metadata_map


# - - - - - - - - - - - - - Card Search - - - - - - - - - - - - -

class CardEmbedder:
    """
    An inference class to handle the creation of full card representations and embeddings.
    It loads all necessary models and data maps once for batched processing.
    """
    def __init__(self, device, cpr_checkpoint_path, partial_map_path, cat_map_path, llm_path, role_class_path, num_types, num_keywords):
        self.device = device
        self.cpr_model = load_card_encoder(cpr_checkpoint_path, num_types, num_keywords, self.device)
        self.partial_map = torch.load(partial_map_path)
        self.cat_map = torch.load(cat_map_path)
        self.all_known_ids = set(self.partial_map.keys())
        self.llm = SentenceTransformer(llm_path, device=self.device)
        self.role_tokenizer = AutoTokenizer.from_pretrained(role_class_path)
        self.role_classifier = AutoModelForSequenceClassification.from_pretrained(role_class_path).to(self.device)
        self.role_classifier.eval()
        self.semantic_dim = 768 # output dim of 'magic-distilbert-base-v1'
        self.role_dim = 16 # number of labels in ROLE_LABELS in text_embeddings.py

    def _assemble_full_representations(self, oids: list):
        """
        Takes a list of oracle_ids and efficiently assembles their full
        925-dimensional representation tensors in a batch.
        """
        batch_partials = [self.partial_map[oid] for oid in oids]
        batch_types = [self.cat_map[oid]["types"] for oid in oids]
        batch_keywords = [self.cat_map[oid]["keywords"] for oid in oids]

        partials_tensor = torch.stack(batch_partials).to(self.device)
        types_tensor = torch.stack(batch_types).to(self.device)
        keywords_tensor = torch.stack(batch_keywords).to(self.device)

        with torch.no_grad():
            cat_embeddings = self.cpr_model.feature_encoder(types_tensor, keywords_tensor)
        
        full_repr_batch = torch.cat([partials_tensor, cat_embeddings], dim=1)
        return full_repr_batch

    def _get_deck_embedding(self, oids: list):
        """
        Gets the final 512-dim embedding for a deck (list of cards).
        """
        if not oids:
            return torch.tensor([])

        full_reprs = self._assemble_full_representations(oids).unsqueeze(0) # Shape: [1, num_cards, 925]
        with torch.no_grad():
            embedding = self.cpr_model.deck_embedding(full_reprs).squeeze(0)
        return embedding

    def _get_prompt_dummy_embedding(self, prompt: str, deck_oids: list):
        """
        Creates a 512-dim embedding representing the prompt's intent, substituting
        both the semantic and role vectors in a prototypical card.
        """
        with torch.no_grad():
            deck_reprs = self._assemble_full_representations(deck_oids)
            dummy_repr = deck_reprs.mean(dim=0)
            
            # partial_repr = [stats, rarity, color, cmc, semantic, roles] -> substitute 'semantic' and 'roles'
            # full_repr = [stats, rarity, color, cmc, semantic, roles, types, keywords]

            # Get fields related to prompt (semantic, roles)
            prompt_semantic_emb = torch.tensor(self.llm.encode(preprocess_text(prompt, remove_reminder=False, mask_name=False)), device=self.device)
            inputs = self.role_tokenizer(preprocess_text(prompt, remove_reminder=False, mask_name=False), return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.role_classifier(**inputs)
            prompt_role_logits = outputs.logits.squeeze(0)

            # Extract averaged filds to be kept 
            keep_dummy_size = len(self.partial_map[deck_oids[0]]) - self.semantic_dim - self.role_dim
            numerical_dummy = dummy_repr[:keep_dummy_size] # (stats, rarity, color, cmc)
            categorical_dummy = dummy_repr[len(self.partial_map[deck_oids[0]]):] # (types, keywords)

            dummy_full_repr = torch.cat([
                numerical_dummy,
                prompt_semantic_emb,
                prompt_role_logits,
                categorical_dummy
            ]).unsqueeze(0)

            # Compute and return embedding of "prompt"
            prompt_intent_embedding = self.cpr_model.card_embedding(dummy_full_repr).squeeze(0)

        return prompt_intent_embedding

    def _get_prompt_direction_vector(self, prompt: str):
        """
        Creates a 512-dim "direction" vector representing the prompt's pure intent
        by building a neutral card scaffold around the prompt's semantic and role features.
        """
        with torch.no_grad():
            # Get semantic and role vectors from the prompt
            prompt_semantic_emb = torch.tensor(self.llm.encode(preprocess_text(prompt, remove_reminder=False, mask_name=False)), device=self.device)
            inputs = self.role_tokenizer(preprocess_text(prompt, remove_reminder=False, mask_name=False), return_tensors="pt", truncation=True, padding=True).to(self.device)
            prompt_role_logits = self.role_classifier(**inputs).logits.squeeze(0)

            # Create a neutral scaffold for the other features
            numerical_feature_size = len(self.partial_map[next(iter(self.all_known_ids))]) - self.semantic_dim - self.role_dim
            neutral_numerical = torch.zeros(numerical_feature_size, device=self.device)

            categorical_feature_size = self.cpr_model.feature_encoder.output_dim
            neutral_categorical = torch.zeros(categorical_feature_size, device=self.device)

            # Assemble the pure prompt card representation
            pure_prompt_repr = torch.cat([
                neutral_numerical,
                prompt_semantic_emb,
                prompt_role_logits,
                neutral_categorical
            ]).unsqueeze(0)

            # Get this vector's direction in the synergy space
            prompt_direction_vector = self.cpr_model.card_embedding(pure_prompt_repr).squeeze(0)
            return F.normalize(prompt_direction_vector, p=2, dim=0)


class CardRetriever:
    def __init__(self, embedder, client, card_dict_path):
        self.embedder = embedder
        self.client = client
        self.name_to_id = torch.load(card_dict_path, weights_only=False)
        self.id_to_name = {val:key for key,val in self.name_to_id.items()}

    def _process_deck_for_search(self, deck_id: int, card_collection=None):
        """
        Given a deck id, returns its embedding and its colors using the CardEmbedder class.
        """
        deck = edh_scraper.archidekt_fetch_deck(deck_id, known_ids=self.embedder.all_known_ids)
        if not deck:
            print(f"Could not fetch or process deck with ID: {deck_id}")
            return None, None, None
        
        decklist_ids = [oid for oid in (deck.commander_ids + deck.mainboard_ids) if oid in self.embedder.all_known_ids]
        if not decklist_ids:
            print("Deck contains no known cards.")
            return None, None, None

        deck_emb_tensor = self.embedder._get_deck_embedding(decklist_ids)
        deck_emb_list = deck_emb_tensor.cpu().tolist()
        
        deck_colors = set()
        if card_collection:
            results = card_collection.get(ids=decklist_ids, include=["metadatas"])
            if results and results.get("metadatas"):
                for metadata in results['metadatas']:
                    color_str = metadata.get("color_identity", ",")
                    card_colors = [color for color in color_str.strip(',').split(',') if color]
                    deck_colors.update(card_colors)
        else:
            print("Failed to access card collection.")
            return [], [], []
        return decklist_ids, deck_emb_list, list(sorted(deck_colors))

    def _build_color_subset_filter(self, colors):
        """
        Builds a ChromaDB '$or' filter to find cards whose color identity is an exact subset of the provided colors (including colorless)
        """
        if isinstance(colors, str):
            color_list = [colors.upper()]
        else:
            color_list = sorted(list({c.upper() for c in colors}))

        allowed_subsets = []
        for r in range(len(color_list) + 1):
            for subset in combinations(color_list, r):
                if not subset:
                    # if processing empty subset, add "C" for colorless
                    allowed_subsets.append(['C'])
                elif 'C' not in subset: # Check wether 'C' was inside original color_list and is getting coupled with other colors
                    allowed_subsets.append(sorted(list(subset)))
        # print(f"allowed colors: {allowed_subsets}")

        # Format each subset into the delimited string format used in the DB
        formatted_subsets = []
        for subset in allowed_subsets:
            formatted_string = f",{','.join(subset)},"
            formatted_subsets.append(formatted_string)
        
        or_conditions = [{"color_identity": {"$eq": s}} for s in formatted_subsets]
        if len(or_conditions) == 1: # If only colorless, just return it
            return or_conditions[0]
        return {"$or": or_conditions} # With more colors, usa an or condition

    def _query_db(self, query_embedding, n:int, colors=None, card_collection=None):
        """
        Takes a deck embedding and returns the top N recommended cards (IDs), optionally filtered by color.
        """
        if card_collection is None:
            print("Failed to access card collection.")
            return []

        if not card_collection:
            print("Client is not available.")
            return None, None

        filter_metadata = {}
        if colors:
            filter_metadata = self._build_color_subset_filter(colors)
        
        try:
            results = card_collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                where=filter_metadata if filter_metadata else None
            )

            if results and results['ids'] and results['ids'][0]:
                return results['ids'][0]
            return []

        except Exception as e:
            print(f"An error occurred during ChromaDB query: {e}")
            return []

    def _mmr_re_rank(self, query_embedding: torch.Tensor, candidate_embeddings: torch.Tensor, candidate_ids: list, lambda_mult: float, top_k: int):
        """
        Performs Maximal Marginal Relevance (MMR) re-ranking on a list of candidate cards. Returns a list of chosen card IDs.
        """
        if not candidate_ids:
            return []

        device = self.embedder.device
        query_emb = query_embedding.to(device)
        candidate_embs = candidate_embeddings.to(device)

        relevance_scores = F.cosine_similarity(query_emb, candidate_embs)
        diversity_scores = F.cosine_similarity(candidate_embs.unsqueeze(1), candidate_embs.unsqueeze(0), dim=2)

        num_candidates = len(candidate_ids)
        remaining_indices = list(range(num_candidates))
        selected_indices = []
        
        first_selection_idx = torch.argmax(relevance_scores).item()
        selected_indices.append(first_selection_idx)
        remaining_indices.remove(first_selection_idx)

        while len(selected_indices) < min(top_k, num_candidates):
            best_score = -float('inf')
            best_idx_to_add = -1
            
            for idx in remaining_indices:
                relevance = relevance_scores[idx]
                redundancy = torch.max(diversity_scores[idx, selected_indices])
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx_to_add = idx

            selected_indices.append(best_idx_to_add)
            remaining_indices.remove(best_idx_to_add)

        return [candidate_ids[i] for i in selected_indices]

    def _stochastic_mmr_re_rank(self, query_embedding: torch.Tensor, candidate_embeddings: torch.Tensor, candidate_ids: list, lambda_mult: float, top_k: int, sample_top_m: int = 5):
        """
        Performs a non-deterministic MMR re-ranking by sampling from the top candidates at each step.
        """
        if not candidate_ids:
            return []

        device = self.embedder.device
        query_emb = query_embedding.to(device)
        candidate_embs = candidate_embeddings.to(device)

        relevance_scores = F.cosine_similarity(query_emb, candidate_embs)
        diversity_scores = F.cosine_similarity(candidate_embs.unsqueeze(1), candidate_embs.unsqueeze(0), dim=2)

        num_candidates = len(candidate_ids)
        
        # Ddictionaries for efficient lookups and removals
        remaining_candidates = {i: cid for i, cid in enumerate(candidate_ids)}
        selected_indices = []
        
        first_selection_idx = torch.argmax(relevance_scores).item()
        selected_indices.append(first_selection_idx)
        del remaining_candidates[first_selection_idx]

        while len(selected_indices) < min(top_k, num_candidates):
            if not remaining_candidates: break

            scores = []
            indices = []
            for idx in remaining_candidates.keys(): # Compute relevance and redundancy for all remaining candidates
                relevance = relevance_scores[idx]
                redundancy = torch.max(diversity_scores[idx, selected_indices])
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                scores.append(mmr_score)
                indices.append(idx)
            
            # Get top m results
            scores_tensor = torch.tensor(scores, device=device)
            top_m = min(sample_top_m, len(scores_tensor))
            top_scores, top_indices_of_scores = torch.topk(scores_tensor, k=top_m)
            # Extract one probabilistically
            probabilities = F.softmax(top_scores, dim=0)
            sampled_idx_within_top_m = torch.multinomial(probabilities, num_samples=1).item()
            # Add selected card to results and remove it from candidates
            chosen_original_index = indices[top_indices_of_scores[sampled_idx_within_top_m]]
            selected_indices.append(chosen_original_index)
            del remaining_candidates[chosen_original_index]

        return [candidate_ids[i] for i in selected_indices]

    def recommend_cards( self,
        deck_id: int,
        db_name: str,
        prompt: str = "",
        n: int = 10,
        mmr_lambda: float = 0.7):
        """
        Main orchestrator for prompt-based recommendations using the CardEmbedder.
        If a prompt is provided, it queries with multiple synergy/prompt blends, pools the results, and uses MMR to select a diverse final set.
        If no prompt is provided, it performs a standard synergy search with MMR for diversity.
        """
        card_collection = self.client.get_collection(name=db_name)
        
        deck_oids, deck_emb_list, deck_colors = self._process_deck_for_search(deck_id, card_collection)
        if deck_emb_list is None:
            print("Failed to process deck.")
            return []
        deck_emb = torch.tensor(deck_emb_list, device=self.embedder.device)

        if prompt and prompt.strip():
            print(f"--- Generating guided recommendations ---")

            prompt_direction_vec = self.embedder._get_prompt_direction_vector(prompt)
            candidate_pool = set()
            
            # Define the spectrum of synergy-to-prompt influence to explore
            alpha_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
            k_per_query = n * 8 # Retrieve more cards, filter them later based on mmr

            for alpha in alpha_values:
                current_query_vector = deck_emb + alpha * prompt_direction_vec
                normalized_query = F.normalize(current_query_vector, p=2, dim=0)
                
                retrieved_ids = self._query_db(
                    query_embedding=normalized_query.cpu().tolist(),
                    n=k_per_query,
                    colors=deck_colors,
                    card_collection=card_collection
                )
                candidate_pool.update(retrieved_ids)

            candidate_ids = list(candidate_pool)
            relevance_query_vector = F.normalize(deck_emb + 0.5 * prompt_direction_vec, p=2, dim=0)
        else:
            print(f"--- Generating synergy recommendations ---")
            candidate_ids = self._query_db(
                    query_embedding=deck_emb.cpu().tolist(),
                    n=n * 10,
                    colors=deck_colors,
                    card_collection=card_collection
                )
            relevance_query_vector = deck_emb

        if not candidate_ids:
            print("No candidates were retrieved.")
            return []

        # print(f"Candidate pool size before filtering: {len(candidate_ids)}")
        deck_oids_set = set(deck_oids)
        candidate_ids = [cid for cid in candidate_ids if cid not in deck_oids_set]
        # print(f"Candidate pool size after filtering out deck cards: {len(candidate_ids)}")

        if not candidate_ids:
            print("No new cards to recommend after filtering.")
            return []

        # Use Maximal Marginal Ranking to diversify results
        candidate_embeddings = torch.tensor(
            card_collection.get(ids=candidate_ids, include=["embeddings"])['embeddings'],
            device=self.embedder.device
            )

        # final_ids = self._mmr_re_rank(
        #     query_embedding=relevance_query_vector,
        #     candidate_embeddings=candidate_embeddings,
        #     candidate_ids=candidate_ids,
        #     lambda_mult=mmr_lambda,
        #     top_k=min(n, len(candidate_ids))
        #     )

        final_ids = self._stochastic_mmr_re_rank(
            query_embedding=relevance_query_vector,
            candidate_embeddings=candidate_embeddings,
            candidate_ids=candidate_ids,
            lambda_mult=mmr_lambda,
            top_k=min(n, len(candidate_ids)),
            sample_top_m=5
            )

        return [self.id_to_name.get(oid, "Unknown Card") for oid in final_ids]


if __name__ == "__main__":
    this = os.path.dirname(__file__)
    data_dir = os.path.join(this, "data")
    cards_metadata = os.path.join(data_dir, "clean_data.json")
    db_path = os.path.join(this, "card_db")
    client = chromadb.PersistentClient(path=db_path)    

    # emb_dict_paths = [os.path.join(this, "data", f"emb_dict_v1_{dataset}_{epochs}_{loss}.pt") for dataset in ["all","div"] for epochs in ["20","200"] for loss in ["nce","3"]]
    # db_names = [f"mtg_cards_v1_{dataset}_{epochs}_{loss}" for dataset in ["all","div"] for epochs in ["20","200"] for loss in ["nce","3"]]
    
    emb_dict_paths = [
        os.path.join(data_dir, "emb_dict_v1_div_20_triplet_s2.pt"),
        os.path.join(data_dir, "emb_dict_v1_all_200_triplet_s2.pt"),
        os.path.join(data_dir, "emb_dict_v1_all_200_nce_s2.pt")
    ]
    db_names = [
        "mtg_cards_v1_div_20_triplet_s2",
        "mtg_cards_v1_all_200_triplet_s2",
        "mtg_cards_v1_all_200_nce_s2",
    ]

    for card_emb_path, db_name in zip(emb_dict_paths, db_names):
        build_and_save_chroma_db(card_emb_path, cards_metadata, client, db_name)