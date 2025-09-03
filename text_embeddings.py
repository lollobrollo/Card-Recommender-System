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
import utils
import re
from more_itertools import chunked 


# ROLE_LABELS = [
#     "artifact sinergy", "blink / flicker", "cloning / copying", "coin flip", "applying counters",
#     "die roll", "enchantment sinergy", "enter the battlefield triggers", "exile",
#     "extra combats", "face-down cards"]

ROLE_LABELS = [
    "Card Draw / Advantage", "Tutor / Search", "Mana Ramp / Source", "Targeted Removal", "Board Wipe / Mass Removal",
    "Interaction / Counterspell", "Threat / Finisher", "Anthem / Power Amplification", "Token Generator",
    "Graveyard Recursion", "Sacrifice Outlet", "Stax / Tax Effect", "Theft / Steal", "Mill", "Discard", "Lifegain"
]

class Paths:
    CLEAN_DATA_JSON = "data/clean_data.json"
    RULES_TXT = "data/mtg_rules.txt"
    # Stage 1 Output
    FOUNDATION_MODEL = "models/magic-distilbert-base-v1"
    # Stage 2 Output
    PSEUDO_LABELED_DATASET = "data/card_roles_dataset.jsonl"
    # Stage 3 Output
    FINAL_CLASSIFIER = "models/card-role-classifier-final"

# Hyperparameters
class Config:
    STAGE1_EPOCHS = 3
    STAGE3_EPOCHS = 4
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    DISTILLATION_TEMP = 2.0 # Temperature for knowledge distillation


# STAGE 1: DOMAIN-ADAPTIVE PRE-TRAINING

def extract_relevant_paragraphs(rules_text: str, min_words: int = 5):
    """
    Parses the text of the MTG Comprehensive Rules and extracts only the
    relevant paragraphs (detailed rules and glossary definitions)
    Args:
        rules_text (str): The full string content of the mtg_rules.txt file.
        min_words (int): The minimum number of words a paragraph must have to be
                         considered a "definition" if it's not a numbered rule.
                         This filters out short, title-like lines.
    Returns:
        List[str]: A list of the cleaned, relevant rule and glossary paragraphs.
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
    if os.path.exists(Paths.FOUNDATION_MODEL):
        print(f"Foundation model already found at '{Paths.FOUNDATION_MODEL}'. Skipping stage 1.")
        return
    if not os.path.exists(Paths.RULES_TXT) or not os.path.exists(Paths.CLEAN_DATA_JSON):
        print("Missing required data files (rules.txt or clean_data.json). Cannot proceed.")
        return
    
    all_texts = []
    with open(Paths.RULES_TXT, 'r', encoding='utf-8') as f:
        rules_content = f.read()
        all_texts.extend(extract_relevant_paragraphs(rules_content))

    with open(Paths.CLEAN_DATA_JSON, 'r', encoding='utf-8') as f:
        for card in ijson.items(f, "item"):
            text = utils.preprocess_oracle_text(text=card.get("oracle_text", ""), card_name=card.get("name", ""))
            if text:
                all_texts.append(text)
    
    print(f"Created a corpus of {len(all_texts)} text documents.")
    hf_dataset = Dataset.from_dict({"text": all_texts})

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=f"{Paths.FOUNDATION_MODEL}_trainer_output",
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

    print(f"Saving foundation model to '{Paths.FOUNDATION_MODEL}'...")
    model.save_pretrained(Paths.FOUNDATION_MODEL)
    tokenizer.save_pretrained(Paths.FOUNDATION_MODEL)


# STAGE 2: ZERO-SHOT PSEUDO-LABELING

def run_stage2_zero_shot_labeling():
    """
    Uses a general-purpose NLI model to automatically label every card with
    role probabilities, creating a dataset for the next stage.
    """
    print("\n--- STAGE 2: Zero-Shot Pseudo-Labeling ---")
    if os.path.exists(Paths.PSEUDO_LABELED_DATASET):
        print(f"Pseudo-labeled dataset already found at '{Paths.PSEUDO_LABELED_DATASET}'. Skipping stage 2.")
        return

    print("Initializing Zero-Shot Classification Pipeline with custom Magic model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

    texts_dict = {}
    with open(Paths.CLEAN_DATA_JSON, 'r', encoding='utf-8') as f:
        for card in ijson.items(f, "item"):
            oid = card.get("oracle_id", "")
            text = utils.preprocess_oracle_text(card.get('oracle_text', ""), card.get("name", ""))
            if oid and text: # check after processing the text
                texts_dict[oid] = text

    magic_template = "The function of this Magic card is {}." # template for the inference model
    items = list(texts_dict.items())
    batch_size = 16
    total_iterations = len(items) // batch_size + bool(len(items) % batch_size)

    print(f"Classifying {len(items)} cards against {len(ROLE_LABELS)} roles in batches of {batch_size}...")
    with open(Paths.PSEUDO_LABELED_DATASET, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(chunked(items, batch_size), total=total_iterations, desc="Generating Pseudo-Labels"):
            batch_texts = [item[1] for item in batch]
            batch_results = classifier(
                batch_texts, 
                candidate_labels=ROLE_LABELS, 
                hypothesis_template=magic_template, 
                multi_label=True
            )

            for i, result in enumerate(batch_results):
                oracle_id = batch[i][0]
                role_scores = {label: score for label, score in zip(result['labels'], result['scores'])}
                f_out.write(json.dumps({"oracle_id": oracle_id, "text": result['sequence'], "scores": role_scores}) + '\n')

    print(f"Pseudo-labeled dataset saved to '{Paths.PSEUDO_LABELED_DATASET}'.")


# STAGE 3: SUPERVISED FINE-TUNING (DISTILLATION)

class DistillationTrainer(Trainer):
    """A custom Trainer for knowledge distillation."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        teacher_probabilities = inputs.get("labels")

        soft_student_log_probs = F.log_softmax(student_logits / Config.DISTILLATION_TEMP, dim=-1)
        soft_teacher_probs = F.softmax(teacher_probabilities / Config.DISTILLATION_TEMP, dim=-1) # idea: diminish teacher overconfidence

        distillation_loss = nn.KLDivLoss(reduction="batchmean")(soft_student_log_probs, soft_teacher_probs)

        return (distillation_loss, student_outputs) if return_outputs else distillation_loss

def run_stage3_supervised_finetuning():
    """
    Fine-tunes the "Magic Foundation Model" on the pseudo-labeled dataset
    using knowledge distillation to create the final, expert Card Role Classifier.
    """
    print("\n--- STAGE 3: Supervised Fine-Tuning (Knowledge Distillation) ---")
    if os.path.exists(Paths.FINAL_CLASSIFIER):
        print(f"Final classifier already found at '{Paths.FINAL_CLASSIFIER}'. Skipping stage 3.")
        return

    hf_dataset = load_dataset('json', data_files=Paths.PSEUDO_LABELED_DATASET, split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(Paths.FOUNDATION_MODEL)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        Paths.FOUNDATION_MODEL,
        num_labels=len(ROLE_LABELS),
        problem_type="multi_label_classification"
    )

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length")
        labels = torch.zeros(len(examples['text']), len(ROLE_LABELS))
        for i, scores in enumerate(examples['scores']):
            for j, role in enumerate(ROLE_LABELS):
                labels[i, j] = scores.get(role, 0.0)
        tokenized['labels'] = labels
        return tokenized
    
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True, remove_columns=['scores', 'oracle_id', 'text'])

    training_args = TrainingArguments(
        output_dir=f"{Paths.FINAL_CLASSIFIER}_trainer_output",
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

    print(f"Saving final card role classifier to '{Paths.FINAL_CLASSIFIER}'...")
    student_model.save_pretrained(Paths.FINAL_CLASSIFIER)
    tokenizer.save_pretrained(Paths.FINAL_CLASSIFIER)



def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    run_stage1_domain_pretraining()
    run_stage2_zero_shot_labeling()
    run_stage3_supervised_finetuning()
    
    print("\n\n--- Pipeline Finished Successfully ---")
    print(f"Card role classifier is saved at: '{Paths.FINAL_CLASSIFIER}'")

if __name__ == "__main__":
    this = os.path.dirname(__file__)
    Paths.CLEAN_DATA_JSON = os.path.join(this, "data", "clean_data.json")
    Paths.RULES_TXT =  os.path.join(this, "data", "mtg_rules.txt")
    Paths.FOUNDATION_MODEL =  os.path.join(this, "models", "magic-distilbert-base-v1")
    Paths.PSEUDO_LABELED_DATASET =  os.path.join(this, "data", "card_roles_dataset.jsonl")
    Paths.FINAL_CLASSIFIER =  os.path.join(this, "models", "card-role-classifier-final")

    main()