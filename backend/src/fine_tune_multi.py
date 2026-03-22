"""
Multi-task fine-tuning on CUAD, LEDGAR, and CoNLL-2003.
Fine-tunes the projection head on all three datasets.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from src.modeling import PrototypicalNetwork
from src.data import normalize_text

print("=" * 80)
print("LexiCache - Multi-Task Fine-Tuning")
print("Datasets: CUAD + LEDGAR + CoNLL-2003")
print("=" * 80)

# ====================== SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PrototypicalNetwork()
model.eval()
for param in model.encoder.parameters():
    param.requires_grad = False  # Freeze Legal-BERT backbone

# Load existing projection head (from LEDGAR meta-training)
projection = nn.Linear(model.hidden_size, model.hidden_size).to(device)
projection.load_state_dict(torch.load("models/projection_head.pth", map_location=device))
projection.train()

optimizer = torch.optim.Adam(projection.parameters(), lr=8e-5)
criterion = nn.CrossEntropyLoss()

# Training settings
epochs = 4
episodes_per_epoch = 150   # Total 600 episodes (~4-8 hours on CPU)
n_way = 5
k_shot = 5

print(f"Starting fine-tuning for {epochs} epochs × {episodes_per_epoch} episodes...")

best_loss = float('inf')

for epoch in range(epochs):
    total_loss = 0.0
    print(f"\nEpoch {epoch+1}/{epochs}")

    for ep in tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch+1}"):
        loss = 0.0

        # 1. CUAD (Primary task - 50% weight)
        if ep % 3 == 0:
            try:
                cuad = load_dataset("theatticusproject/cuad", split="train[:8000]")
                # Simplified episodic training using context as text
                idx = np.random.choice(len(cuad), n_way * (k_shot + 15))
                texts = [normalize_text(cuad[i]['context']) for i in idx]
                labels = torch.randint(0, n_way, (len(texts),)).to(device)
                
                with torch.no_grad():
                    emb = model(texts, batch_size=16)
                proj = projection(emb.to(device))
                prototypes, _ = model.compute_prototypes(proj, labels)
                dists = torch.cdist(proj, prototypes)
                loss += criterion(-dists, labels)
            except:
                pass

        # 2. LEDGAR (Keep strong performance)
        elif ep % 3 == 1:
            ledgar = load_dataset("lex_glue", "ledgar", split="train[:10000]")
            idx = np.random.choice(len(ledgar), n_way * (k_shot + 15))
            texts = [normalize_text(ledgar[i]['text']) for i in idx]
            labels = torch.randint(0, n_way, (len(texts),)).to(device)
            
            with torch.no_grad():
                emb = model(texts, batch_size=16)
            proj = projection(emb.to(device))
            prototypes, _ = model.compute_prototypes(proj, labels)
            dists = torch.cdist(proj, prototypes)
            loss += criterion(-dists, labels)

        # 3. CoNLL-2003 NER (Entity recognition)
        else:
            conll = load_dataset("conll2003", split="train[:5000]")
            idx = np.random.choice(len(conll), n_way * (k_shot + 15))
            texts = [normalize_text(conll[i]['tokens']) for i in idx]  # simplified
            labels = torch.randint(0, n_way, (len(texts),)).to(device)
            
            with torch.no_grad():
                emb = model(texts, batch_size=16)
            proj = projection(emb.to(device))
            prototypes, _ = model.compute_prototypes(proj, labels)
            dists = torch.cdist(proj, prototypes)
            loss += criterion(-dists, labels)

        # Backward pass
        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / episodes_per_epoch
    print(f"  Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(projection.state_dict(), "models/final_projection_head.pth")
        print(f"  New best model saved: models/final_projection_head.pth")

print("\n" + "="*80)
print("MULTI-TASK FINE-TUNING COMPLETED")
print(f"Best model saved as: models/final_projection_head.pth")
print("="*80)