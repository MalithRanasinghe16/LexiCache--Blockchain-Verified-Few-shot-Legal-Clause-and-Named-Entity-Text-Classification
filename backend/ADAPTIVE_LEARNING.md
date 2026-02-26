# LexiCache Adaptive Meta-Learning System

## Overview

LexiCache now implements a **professional adaptive meta-learning system** that supports:

- ✅ **All 41 CUAD Standard Clause Types** (not hardcoded in the model)
- ✅ **Online Few-Shot Learning** (learns from user feedback in real-time)
- ✅ **Persistent Support Set** (knowledge saved across sessions)
- ✅ **Hybrid Classification** (keywords + meta-learned model)
- ✅ **Unknown Clause Detection** (flags low-confidence predictions)
- ✅ **Confidence-Based Routing** (automatic quality assessment)

## Architecture

### 3-Tier Classification Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Contract Clause                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   1. Keyword Heuristics (Fast)         │
         │   - 41 CUAD types with keywords        │
         │   - Interpretable, no training needed  │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   2. Meta-Learned Model (Adaptive)     │
         │   - Prototypical network embeddings    │
         │   - Learns from user feedback          │
         │   - Persistent support set             │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │   3. Confidence-Based Decision         │
         │   High (>75%): Trust prediction        │
         │   Medium (55-75%): Show with caution   │
         │   Low (<55%): Flag for user review     │
         └────────────────────────────────────────┘
```

## All 41 CUAD Clause Types

### Core Agreement Terms

- Document Name
- Parties
- Agreement Date
- Effective Date
- Expiration Date

### Financial & Payment

- Payment Terms
- Cap on Liability
- Liquidated Damages
- Revenue/Profit Sharing
- Price Restrictions
- Minimum Commitment
- Volume Restriction

### Liability & Risk

- Limitation of Liability
- Indemnification
- Warranty Duration
- Insurance

### Termination & Renewal

- Termination
- Termination for Convenience
- Renewal Term
- Notice Period to Terminate Renewal
- Post-Termination Services

### Restrictions & Competition

- Non-Compete
- Exclusivity
- No-Solicit of Customers
- No-Solicit of Employees
- Non-Disparagement

### IP & Confidentiality

- Intellectual Property
- IP Ownership Assignment
- Joint IP Ownership
- License Grant
- Confidentiality

### Change & Updates

- Change of Control
- Anti-Assignment
- Covenant Not to Sue

### Legal & Governance

- Governing Law
- Dispute Resolution
- Jurisdiction
- Notice
- Force Majeure
- Severability
- Entire Agreement
- Amendment
- Waiver

### Other

- Third Party Beneficiary
- Audit Rights
- ROFR/ROFO/ROFN
- Most Favored Nation

## API Endpoints

### 1. Get Statistics

```bash
GET /statistics
```

Response:

```json
{
  "status": "success",
  "statistics": {
    "total_examples": 25,
    "unique_types": 8,
    "known_cuad_types": 41,
    "label_distribution": {
      "Governing Law": 5,
      "Termination": 3,
      "Custom Clause Type": 2
    },
    "device": "cpu"
  }
}
```

### 2. Submit Feedback (Online Learning)

```bash
POST /feedback
Content-Type: application/json

{
  "clause_text": "This agreement shall be governed by the laws of California",
  "correct_label": "Governing Law",
  "original_prediction": "Unknown Clause",
  "confidence": 0.45
}
```

Response:

```json
{
  "status": "success",
  "message": "Learned 'Governing Law' successfully",
  "model_stats": {
    "total_examples": 26,
    "unique_types": 8
  }
}
```

### 3. Get Clause Types

```bash
GET /clause-types
```

Response:

```json
{
  "status": "success",
  "cuad_types": ["Governing Law", "Termination", ...],
  "learned_types": ["Governing Law", "Custom Type", ...],
  "total_known_types": 43
}
```

## Online Learning Workflow

### For Developers

1. **User encounters unknown clause**

   ```python
   # Prediction comes back with low confidence
   {
     "clause_type": "Unknown Clause",
     "confidence": 0.42,
     "needs_review": true
   }
   ```

2. **UI prompts user for correct label**

   ```typescript
   // Frontend shows modal/input
   const correctLabel = await promptUser(clause.span);
   ```

3. **Submit feedback to API**

   ```typescript
   const response = await fetch("/feedback", {
     method: "POST",
     body: JSON.stringify({
       clause_text: clause.span,
       correct_label: correctLabel,
     }),
   });
   ```

4. **Model learns immediately**
   - Embedding added to support set
   - Future similar clauses classified correctly
   - Knowledge persisted to disk

### For Researchers (Thesis)

**Why This Approach Works:**

1. **No Supervised Training on 41 Types**
   - Only meta-trained on few-shot learning task
   - Generalizes to new types without retraining
   - Realistic for real-world deployment

2. **Hybrid System**
   - Keywords handle common/obvious cases (fast, no training)
   - Meta-learning handles edge cases (adaptive, learns from user)
   - Best of both worlds

3. **Active Learning**
   - System knows when it's uncertain
   - Only asks user for help when needed
   - Each feedback improves future predictions

4. **Persistent Knowledge**
   - Support set saved to `support_set.pkl`
   - Survives server restarts
   - Accumulates domain knowledge over time

## Files

- `src/ml_model.py` - Core adaptive model
- `support_set.pkl` - Persistent support set (auto-created)
- `main.py` - FastAPI endpoints

## Confidence Thresholds

```python
high_confidence_threshold = 0.75   # Trust completely
medium_confidence_threshold = 0.55  # Show but mark uncertain
low_confidence_threshold = 0.40     # Flag for user review
```

## Example: Learning Session

```
Session 1: Upload contract
├─ 10 clauses detected
├─ 2 marked as "Unknown Clause" (conf: 0.42, 0.38)
└─ User provides correct labels
    ├─ "Audit Rights" ✓ Learned
    └─ "ROFR" ✓ Learned

Session 2: Upload similar contract (same domain)
├─ 12 clauses detected
├─ "Audit Rights" now detected at 0.82 confidence ✓
├─ "ROFR" now detected at 0.79 confidence ✓
└─ Model improved without any retraining!
```

## Advantages for Thesis

✅ **Realistic**: No unrealistic assumption of labeled data for all 41 types  
✅ **Scalable**: Can learn 50, 100, or more types as needed  
✅ **Interactive**: Human-in-the-loop improves accuracy  
✅ **Practical**: Works in production without ML engineer  
✅ **Research Novel**: Combines meta-learning + active learning + persistence

## Future Enhancements

1. **Confidence Calibration**: Better uncertainty estimation
2. **Active Learning Strategy**: Smart selection of which samples to ask about
3. **Multi-Model Ensemble**: Combine multiple meta-learners
4. **Federated Learning**: Learn from multiple user deployments
5. **Explanation Module**: Show why model made prediction

## Citation

If using this system in academic work:

```
LexiCache: Adaptive Meta-Learning for Legal Clause Classification
Few-shot learning with online adaptation and persistent knowledge
CUAD Dataset: https://www.atticusprojectai.org/cuad
```
