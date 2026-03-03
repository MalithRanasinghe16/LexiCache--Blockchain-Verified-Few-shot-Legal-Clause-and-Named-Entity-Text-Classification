
"""
Few-shot experiments runner for LexiCache.
Loads LEDGAR, runs prototypical episodes, and reports macro F1.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from datasets import load_dataset
from src.modeling import PrototypicalNetwork
from src.data import normalize_text  

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


# Meta-Training: Episodic training of projection head on top of frozen encoder
def train_prototypical_meta(
    n_way: int = 5,
    k_shot: int = 5,
    n_episodes_train: int = 500,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_path: str = "projection_head.pth"
):
    """
    Meta-train projection head with correct label remapping per episode.
    """
    print(f"Starting meta-training: {n_way}-way {k_shot}-shot | {n_episodes_train} episodes | {epochs} epochs")
    
    dataset = load_dataset("lex_glue", "ledgar")
    train_texts = dataset['train']['text']
    train_labels = np.array(dataset['train']['label'])  # numpy for faster indexing
    
    model = PrototypicalNetwork("nlpaueb/legal-bert-base-uncased")
    model.eval()  # freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    projection = nn.Linear(model.hidden_size, model.hidden_size).to(model.device)
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = n_episodes_train // epochs
        
        for ep in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            # Sample n_way unique global classes
            global_classes = np.random.choice(np.unique(train_labels), n_way, replace=False)
            
            # Create mapping: global label → local index 0 to n_way-1
            global_to_local = {g: i for i, g in enumerate(global_classes)}
            
            support_idx, query_idx = [], []
            support_local_labels, query_local_labels = [], []
            
            for local_id, global_c in enumerate(global_classes):
                c_idx = np.where(train_labels == global_c)[0]
                np.random.shuffle(c_idx)
                support_idx.extend(c_idx[:k_shot])
                query_idx.extend(c_idx[k_shot:k_shot + 15])
                
                # Assign local labels
                support_local_labels.extend([local_id] * k_shot)
                query_local_labels.extend([local_id] * len(c_idx[k_shot:k_shot + 15]))
            
            # Prepare texts (normalized)
            support_texts = [normalize_text(train_texts[i]) for i in support_idx]
            query_texts   = [normalize_text(train_texts[i]) for i in query_idx]
            
            # Convert to tensors (local labels!)
            support_labels_local = torch.tensor(support_local_labels).to(model.device)
            query_labels_local   = torch.tensor(query_local_labels).to(model.device)
            
            # Forward
            with torch.no_grad():
                support_emb = model(support_texts, batch_size=batch_size)
                query_emb   = model(query_texts,   batch_size=batch_size)
            
            support_proj = projection(support_emb.to(model.device))
            query_proj   = projection(query_emb.to(model.device))
            
            # Prototypes using local labels
            prototypes, _ = model.compute_prototypes(support_proj, support_labels_local)
            
            # Distances
            dists = torch.cdist(query_proj, prototypes)  # [n_query, n_way]
            
            # Loss: now targets are 0..n_way-1
            loss = criterion(-dists, query_labels_local)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(projection.state_dict(), save_path)
            print(f"  Saved better projection: {save_path} (loss {avg_loss:.4f})")
    
    print("\nMeta-training completed.")
    print(f"Best projection saved to: {save_path}")
    
    # Quick message: now evaluate with this projection
    print("\nNext step: load projection and run evaluation with projected embeddings.")
    return projection

def evaluate_with_trained_projection(
    n_way: int = 5,
    k_shot: int = 5,
    n_episodes: int = 50,
    projection_path: str = "projection_head.pth",
    apply_normalization: bool = True
):
    """
    Evaluate few-shot performance using the trained projection head.
    Compares against frozen baseline.
    """
    print(f"Evaluating with trained projection: {n_way}-way {k_shot}-shot | {n_episodes} episodes")
    
    dataset = load_dataset("lex_glue", "ledgar")
    train_texts = dataset['train']['text']
    train_labels = np.array(dataset['train']['label'])
    
    model = PrototypicalNetwork("nlpaueb/legal-bert-base-uncased")
    model.eval()
    
    # Load trained projection
    projection = nn.Linear(model.hidden_size, model.hidden_size).to(model.device)
    projection.load_state_dict(torch.load(projection_path, map_location=model.device))
    projection.eval()
    
    all_preds, all_true = [], []
    
    for episode in tqdm(range(n_episodes), desc="Evaluation Episodes"):
        global_classes = np.random.choice(np.unique(train_labels), n_way, replace=False)
        global_to_local = {g: i for i, g in enumerate(global_classes)}
        
        support_idx, query_idx = [], []
        support_local, query_local = [], []
        
        for local_id, global_c in enumerate(global_classes):
            c_idx = np.where(train_labels == global_c)[0]
            np.random.shuffle(c_idx)
            support_idx.extend(c_idx[:k_shot])
            query_idx.extend(c_idx[k_shot:k_shot + 15])
            
            support_local.extend([local_id] * k_shot)
            query_local.extend([local_id] * len(c_idx[k_shot:k_shot + 15]))
        
        support_texts = [normalize_text(train_texts[i]) for i in support_idx] if apply_normalization else [train_texts[i] for i in support_idx]
        query_texts   = [normalize_text(train_texts[i]) for i in query_idx]   if apply_normalization else [train_texts[i] for i in query_idx]
        
        support_labels_local = torch.tensor(support_local).to(model.device)
        query_labels_local   = torch.tensor(query_local).to(model.device)
        
        with torch.no_grad():
            support_emb = model(support_texts, batch_size=16)
            query_emb   = model(query_texts,   batch_size=16)
            
            support_proj = projection(support_emb.to(model.device))
            query_proj   = projection(query_emb.to(model.device))
        
        prototypes, _ = model.compute_prototypes(support_proj, support_labels_local)
        dists = torch.cdist(query_proj, prototypes)
        preds = torch.argmin(dists, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(query_labels_local.cpu().numpy())
    
    macro_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"\nMacro F1 with trained projection ({n_way}-way {k_shot}-shot): {macro_f1:.4f}")
    print(f"(Previous frozen baseline was ~0.7277)")

if __name__ == "__main__":
    # Option A: Run evaluation only 
    # run_few_shot_ledgar(n_way=5, k_shot=5, n_episodes=50, apply_normalization=True)
    
    # Option B: Run meta-training 
    # train_prototypical_meta(
    #     n_way=5,
    #     k_shot=5,
    #     n_episodes_train=1000,      
    #     epochs=5,
    #     batch_size=16,
    #     lr=1e-4,
    #     save_path="projection_head.pth"
    # )


    
    # Evaluate with the saved projection
    evaluate_with_trained_projection(
        n_way=5,
        k_shot=5,
        n_episodes=50,
        projection_path="projection_head.pth",
        apply_normalization=True
    )
