"""
Few-shot Experiments Runner
Loads data → runs prototypical episodes → reports avg F1 over 1000 episodes.
"""
# import sys
# import os

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)


from datasets import load_dataset
from src.modeling import PrototypicalNetwork
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

def run_few_shot_ledgar(n_way=5, k_shot=5, n_episodes=500):
    print(f"Running {n_episodes} episodes: {n_way}-way {k_shot}-shot on LEDGAR")
    
    dataset = load_dataset("lex_glue", "ledgar")
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']

    model = PrototypicalNetwork("nlpaueb/legal-bert-base-uncased")
    model.eval()  # no fine-tuning yet – pure few-shot embedding

    all_preds, all_true = [], []

    for episode in tqdm(range(n_episodes)):
        # Sample classes
        classes = np.random.choice(np.unique(train_labels), n_way, replace=False)
        
        support_idx, query_idx = [], []
        for c in classes:
            c_idx = np.where(train_labels == c)[0]
            np.random.shuffle(c_idx)
            support_idx.extend(c_idx[:k_shot])
            query_idx.extend(c_idx[k_shot:k_shot+15])  # 15 queries per class

        support_texts = [train_texts[i] for i in support_idx]
        support_labels = torch.tensor([train_labels[i] for i in support_idx])
        query_texts = [train_texts[i] for i in query_idx]
        query_labels = torch.tensor([train_labels[i] for i in query_idx])

        with torch.no_grad():
            support_emb = model(support_texts)
            query_emb = model(query_texts)

        prototypes, proto_labels = model.compute_prototypes(support_emb, support_labels)
        preds, _ = model.classify(query_emb, prototypes)

        # Map preds back to original labels
        pred_labels = proto_labels[preds]

        all_preds.extend(pred_labels.cpu().numpy())
        all_true.extend(query_labels.numpy())

    macro_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"Macro F1 over {n_episodes} episodes: {macro_f1:.4f}")

if __name__ == "__main__":
    run_few_shot_ledgar(n_way=5, k_shot=5, n_episodes=200)  # start small