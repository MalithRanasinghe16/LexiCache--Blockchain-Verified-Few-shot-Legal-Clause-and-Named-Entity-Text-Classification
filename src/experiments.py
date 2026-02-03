
"""
Few-shot Experiments Runner for LexiCache

Loads LEDGAR → runs prototypical episodes → reports macro F1.
Baseline: 0.726 macro F1 (5-way 5-shot, 200 episodes) on frozen Legal-BERT.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from datasets import load_dataset
from src.modeling import PrototypicalNetwork
from src.data import normalize_text  # Your normalization function

def run_few_shot_ledgar(n_way=5, k_shot=5, n_episodes=50, apply_normalization=True):
    print(f"Running {n_episodes} episodes: {n_way}-way {k_shot}-shot on LEDGAR")
    
    dataset = load_dataset("lex_glue", "ledgar")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']

    model = PrototypicalNetwork("nlpaueb/legal-bert-base-uncased")

    all_preds, all_true = [], []

    for episode in tqdm(range(n_episodes), desc="Episodes"):
        classes = np.random.choice(np.unique(train_labels), n_way, replace=False)
        
        support_idx, query_idx = [], []
        for c in classes:
            c_idx = np.where(np.array(train_labels) == c)[0]
            np.random.shuffle(c_idx)
            support_idx.extend(c_idx[:k_shot])
            query_idx.extend(c_idx[k_shot:k_shot + 15])  # ~15 queries/class

        support_texts = [train_texts[i] for i in support_idx]
        query_texts = [train_texts[i] for i in query_idx]

        # Apply text normalization
        if apply_normalization:
            support_texts = [normalize_text(t) for t in support_texts]
            query_texts = [normalize_text(t) for t in query_texts]

        support_labels = torch.tensor([train_labels[i] for i in support_idx])
        query_labels = torch.tensor([train_labels[i] for i in query_idx])

        support_emb = model(support_texts, batch_size=16)
        query_emb = model(query_texts, batch_size=16)

        prototypes, proto_labels = model.compute_prototypes(support_emb, support_labels)
        preds, _ = model.classify(query_emb, prototypes)

        pred_labels = proto_labels[preds]

        all_preds.extend(pred_labels.numpy())
        all_true.extend(query_labels.numpy())

    macro_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"\nMacro F1 over {n_episodes} episodes ({n_way}-way {k_shot}-shot): {macro_f1:.4f}")

if __name__ == "__main__":
    run_few_shot_ledgar(n_way=5, k_shot=5, n_episodes=50, apply_normalization=True)