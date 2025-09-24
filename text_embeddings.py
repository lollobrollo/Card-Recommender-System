"""
    Author : Lorenzo Bortolussi
    Year : 2024/2025
    This code is part of the implementation of the project developed for my Thesis in Artificial Intelligence and Data Analytics.
"""

import os
import json
import ijson
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_dataset, Dataset
from utils import preprocess_text
import re
from more_itertools import chunked
import random
import pandas as pd
import matplotlib.pyplot as plt

# role_labels = [
#     "artifact sinergy", "blink / flicker", "cloning / copying", "coin flip", "applying counters",
#     "die roll", "enchantment sinergy", "enter the battlefield triggers", "exile",
#     "extra combats", "face-down cards"]

role_labels = [
    "Card Draw / Advantage", "Tutor / Search", "Mana Ramp / Source", "Targeted Removal", "Board Wipe / Mass Removal",
    "Interaction / Counterspell", "Threat / Finisher", "Anthem / Power Amplification", "Token Generator",
    "Graveyard Recursion", "Sacrifice Outlet", "Stax / Tax Effect", "Theft / Steal", "Mill", "Discard", "Lifegain"
]

class Paths:
    clean_data_json = "data/clean_data.json"
    rules_txt = "data/mtg_rules.txt"
    scraped_articles_txt = "data/scraped_articles.txt"
    # Stage 1 Output
    fundation_model = "models/magic-distilbert-base-v1"
    # Stage 2 Output
    pseudo_labeled_dataset = "data/card_roles_dataset.jsonl"
    # Stage 3 Output
    final_classifier = "models/card-role-classifier-final"

# Hyperparameters
class Config:
    STAGE1_EPOCHS = 10
    STAGE3_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    DISTILLATION_TEMP = 2.0 # Temperature for knowledge distillation
    ALPHA = 0.4 # Trade-off beween KL-loss and BCE for stage 3 training


# STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING

def extract_relevant_paragraphs(rules_text:str, min_words:int = 5):
    """
    Parses the text of the MTG Comprehensive Rules and extracts only the relevant paragraphs (detailed rules and glossary definitions)
    """
    rule_pattern = re.compile(r'^\d{1,3}\.\d+')
    paragraphs = rules_text.split('\n\n')
    
    relevant_paragraphs = []
    for p in paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        
        is_numbered_rule = bool(rule_pattern.match(p_stripped))
        is_long_enough = len(p_stripped.split()) > min_words

        if is_numbered_rule or is_long_enough:
            relevant_paragraphs.append(p_stripped)

    return relevant_paragraphs


def run_stage1_domain_pretraining():
    """
    Fine-tunes a DistilBERT model on a mixed corpus of MTG Rules and Oracle Texts
    using Masked Language Modeling (MLM). This creates a "Magic Foundation Model".
    """
    print("\n--- STAGE 1: Domain-Adaptive Pre-training (MLM) ---")
    if os.path.exists(Paths.fundation_model):
        print(f"Foundation model already found at '{Paths.fundation_model}'. Skipping stage 1.")
        return
    if not os.path.exists(Paths.rules_txt) or not os.path.exists(Paths.clean_data_json):
        print("Missing required data files (rules.txt or clean_data.json). Cannot proceed.")
        return
    
    all_texts = []
    with open(Paths.rules_txt, "r", encoding="utf-8") as f:
        rules_content = f.read()
        relevant_paragraphs = extract_relevant_paragraphs(rules_content)
        all_texts.extend([preprocess_text(text, mask_name=False) for text in relevant_paragraphs])

    with open(Paths.clean_data_json, "r", encoding="utf-8") as f:
        for card in ijson.items(f, "item"):
            text = preprocess_text(text=card.get("oracle_text", ""), card_name=card.get("name", ""))
            if text:
                all_texts.append(text)

    with open(Paths.scraped_articles_txt, "r", encoding="utf-8") as f:
        scraped_lines = [preprocess_text(line.strip(), mask_name=False) for line in f if line.strip()]
        all_texts.extend(scraped_lines)
    random.shuffle(all_texts)

    print(f"Created a corpus of {len(all_texts)} text documents.")
    hf_dataset = Dataset.from_dict({"text": all_texts})

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def build_vocab(texts, min_freq=5):
        vocab = {}
        for text in texts:
            for token in text.split():
                vocab[token] = vocab.get(token, 0) + 1
        return [w for w, c in vocab.items() if c >= min_freq]

    domain_vocab = build_vocab(all_texts)
    tokenizer.add_tokens(domain_vocab) # Extend tokenizer with context specific tokens
    print(f"Extended tokenizer with {len(domain_vocab)} MTG-specific tokens.")

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    model.resize_token_embeddings(len(tokenizer)) # Resize embeddings to account for new tokens

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=f"{Paths.fundation_model}_trainer_output",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.STAGE1_EPOCHS,
        weight_decay=0.01,
        logging_steps=500,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("Starting MLM fine-tuning...")
    trainer.train()

    print(f"Saving foundation model to '{Paths.fundation_model}'...")
    model.save_pretrained(Paths.fundation_model)
    tokenizer.save_pretrained(Paths.fundation_model)


# STAGE 2: ZERO-SHOT PSEUDO-LABELING

def run_stage2_zero_shot_labeling():
    """
    Uses a general-purpose NLI model to automatically label every card with
    role probabilities, creating a dataset for the next stage.
    """
    print("\n--- STAGE 2: Zero-Shot Pseudo-Labeling ---")
    if os.path.exists(Paths.pseudo_labeled_dataset):
        print(f"Pseudo-labeled dataset already found at '{Paths.pseudo_labeled_dataset}'. Skipping stage 2.")
        return

    print("Initializing Zero-Shot Classification Pipeline with custom Magic model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    texts_dict = {}
    with open(Paths.clean_data_json, 'r', encoding='utf-8') as f:
        for card in ijson.items(f, "item"):
            oid = card.get("oracle_id", "")
            text = preprocess_text(text=card.get('oracle_text', ""), card_name=card.get("name", ""))
            if oid and text: # check after processing the text
                texts_dict[oid] = text

    magic_template = "The function of this Magic card is {}." # template for the inference model
    items = list(texts_dict.items())
    batch_size = 16
    total_iterations = len(items) // batch_size + bool(len(items) % batch_size)

    print(f"Classifying {len(items)} cards against {len(role_labels)} roles in batches of {batch_size}...")
    with open(Paths.pseudo_labeled_dataset, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(chunked(items, batch_size), total=total_iterations, desc="Generating Pseudo-Labels"):
            batch_texts = [item[1] for item in batch]
            batch_results = classifier(
                batch_texts, 
                candidate_labels=role_labels, 
                hypothesis_template=magic_template, 
                multi_label=True
            )

            
            CONF_THRESHOLD = 0.2
            for i, result in enumerate(batch_results):
                oracle_id = batch[i][0]
                role_scores = {}
                for label, score in zip(result['labels'], result['scores']):
                    if score < CONF_THRESHOLD:
                        score *= 0.1 # labels with low confidence get their score lowered further in magnitude (lower importance in stage 3)
                    role_scores[label] = score

                if role_scores:
                    f_out.write(json.dumps({
                        "oracle_id": oracle_id,
                        "text": result['sequence'],
                        "scores": role_scores
                    }) + '\n')

    print(f"Pseudo-labeled dataset saved to '{Paths.pseudo_labeled_dataset}'.")


# STAGE 3: SUPERVISED FINE-TUNING (DISTILLATION)

class DistillationTrainer(Trainer):
    """A custom Trainer for knowledge distillation."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # kwargs is here because it accepts num_items_in_batch
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        teacher_probs = inputs.get("labels")

        # KL divergence (softened distributions)
        soft_student_log_probs = F.log_softmax(student_logits / Config.DISTILLATION_TEMP, dim=-1)
        soft_teacher_probs = F.softmax(teacher_probs / Config.DISTILLATION_TEMP, dim=-1) # idea: diminish teacher overconfidence
        distill_loss = nn.KLDivLoss(reduction="batchmean")(soft_student_log_probs, soft_teacher_probs)

        # BCE on teacher-provided scores 
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, teacher_probs)

        loss = Config.ALPHA * distill_loss + (1 - Config.ALPHA) * hard_loss
        return (loss, student_outputs) if return_outputs else loss


def run_stage3_supervised_finetuning():
    """
    Fine-tunes the "Magic Foundation Model" on the pseudo-labeled dataset
    using knowledge distillation to create the final, expert Card Role Classifier.
    """
    print("\n--- STAGE 3: Supervised Fine-Tuning (Knowledge Distillation) ---")
    if os.path.exists(Paths.final_classifier):
        print(f"Final classifier already found at '{Paths.final_classifier}'. Skipping stage 3.")
        return

    hf_dataset = load_dataset('json', data_files=Paths.pseudo_labeled_dataset, split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(Paths.fundation_model)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        Paths.fundation_model,
        num_labels=len(role_labels),
        problem_type="multi_label_classification"
    )

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length")
        labels = torch.zeros(len(examples['text']), len(role_labels))
        for i, scores in enumerate(examples['scores']):
            for j, role in enumerate(role_labels):
                labels[i, j] = scores.get(role, 0.0)
        tokenized['labels'] = labels
        return tokenized
    
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=['scores', 'oracle_id', 'text'])

    training_args = TrainingArguments(
        output_dir=f"{Paths.final_classifier}_trainer_output",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.STAGE3_EPOCHS,
        weight_decay=0.01,
        logging_steps=250,
    )
    trainer = DistillationTrainer(
        model=student_model, args=training_args, train_dataset=tokenized_dataset
    )
    
    print("Starting fine-tuning with knowledge distillation...")
    trainer.train()

    print(f"Saving final card role classifier to '{Paths.final_classifier}'...")
    student_model.save_pretrained(Paths.final_classifier)
    tokenizer.save_pretrained(Paths.final_classifier)


def check_corpus():
    print("Analyzing corpus sequence lengths...")
    all_texts = []
    with open(Paths.rules_txt, "r", encoding="utf-8") as f:
        rules_content = f.read()
        relevant_paragraphs = extract_relevant_paragraphs(rules_content)
        all_texts.extend([preprocess_text(text, mask_name=False) for text in relevant_paragraphs])

    with open(Paths.clean_data_json, "r", encoding="utf-8") as f:
        for card in ijson.items(f, "item"):
            text = preprocess_text(text=card.get("oracle_text", ""), card_name=card.get("name", ""))
            if text:
                all_texts.append(text)

    with open(Paths.scraped_articles_txt, "r", encoding="utf-8") as f:
        scraped_lines = [preprocess_text(line.strip(), mask_name=False) for line in f if line.strip()]
        all_texts.extend(scraped_lines)
    random.shuffle(all_texts)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    lengths = [len(tokenizer.encode(text)) for text in all_texts]

    pd.Series(lengths).hist(bins=50)
    plt.title("Distribution of Tokenized Sequence Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.show()

    print(f"Sequence Length Stats:\n{pd.Series(lengths).describe(percentiles=[0.5, 0.8, 0.9, 0.95, 0.99])}")



def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    run_stage1_domain_pretraining()
    run_stage2_zero_shot_labeling()
    run_stage3_supervised_finetuning()
    
    print("\n\n--- Pipeline Finished Successfully ---")
    print(f"Card role classifier is saved at: '{Paths.final_classifier}'")

if __name__ == "__main__":
    this = os.path.dirname(__file__)
    Paths.clean_data_json = os.path.join(this, "data", "clean_data.json")
    Paths.rules_txt = os.path.join(this, "data", "mtg_rules.txt")
    Paths.fundation_model = os.path.join(this, "models", "magic-distilbert-base-v1")
    Paths.pseudo_labeled_dataset = os.path.join(this, "data", "card_roles_dataset.jsonl")
    Paths.final_classifier = os.path.join(this, "models", "card-role-classifier-final")
    Paths.scraped_articles_txt = os.path.join(this, "data", "scraped_articles.txt")

    main()

    #check_corpus()