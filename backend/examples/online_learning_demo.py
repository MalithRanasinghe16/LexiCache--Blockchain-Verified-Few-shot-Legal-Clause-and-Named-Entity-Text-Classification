"""
"""LexiCache online learning demo.
Demonstrates how the adaptive meta-learning system works.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def get_statistics():
    """Get current model statistics"""
    response = requests.get(f"{BASE_URL}/statistics")
    return response.json()

def submit_feedback(clause_text, correct_label, original_prediction=None, confidence=None):
    """Submit user feedback to improve the model"""
    data = {
        "clause_text": clause_text,
        "correct_label": correct_label,
        "original_prediction": original_prediction,
        "confidence": confidence
    }
    response = requests.post(f"{BASE_URL}/feedback", json=data)
    return response.json()

def predict_text(text):
    """Get predictions for text"""
    response = requests.post(f"{BASE_URL}/predict-text", json={"text": text})
    return response.json()

def get_clause_types():
    """Get all known clause types"""
    response = requests.get(f"{BASE_URL}/clause-types")
    return response.json()


if __name__ == "__main__":
    print_section("LexiCache Adaptive Learning Demo")
    
    # 1. Check initial statistics
    print_section("Step 1: Initial Model Statistics")
    stats = get_statistics()
    print(json.dumps(stats, indent=2))
    
    # 2. Get all known clause types
    print_section("Step 2: Known Clause Types")
    clause_types = get_clause_types()
    print(f"CUAD Standard Types: {len(clause_types['data']['cuad_types'])}")
    print(f"Learned Types: {len(clause_types['data']['learned_types'])}")
    print(f"Total: {clause_types['data']['total_known_types']}")
    
    # 3. Example: Submit feedback for unknown clause
    print_section("Step 3: Learning from User Feedback")
    
    examples = [
        {
            "text": "The Supplier shall maintain comprehensive general liability insurance coverage of at least $5,000,000.",
            "label": "Insurance",
            "description": "Insurance clause"
        },
        {
            "text": "Buyer shall have the right to audit Seller's books and records with 30 days written notice.",
            "label": "Audit Rights",
            "description": "Audit rights clause"
        },
        {
            "text": "Either party may terminate this agreement for convenience upon 90 days written notice.",
            "label": "Termination for Convenience",
            "description": "Termination for convenience"
        },
        {
            "text": "Company grants Customer a non-exclusive, worldwide license to use the Software.",
            "label": "License Grant",
            "description": "License grant clause"
        },
        {
            "text": "This Agreement shall automatically renew for successive one-year terms unless terminated.",
            "label": "Renewal Term",
            "description": "Automatic renewal"
        }
    ]
    
    for idx, example in enumerate(examples, 1):
        print(f"\n{idx}. Teaching: {example['description']}")
        print(f"   Text: {example['text'][:80]}...")
        
        result = submit_feedback(
            clause_text=example['text'],
            correct_label=example['label']
        )
        
        if result['status'] == 'success':
            print(f"   Successfully learned: {example['label']}")
            print(f"   Total examples: {result['model_stats']['total_examples']}")
        else:
            print(f"   Failed to learn")
    
    # 4. Check updated statistics
    print_section("Step 4: Updated Model Statistics")
    stats = get_statistics()
    print(f"Total Examples: {stats['statistics']['total_examples']}")
    print(f"Unique Types: {stats['statistics']['unique_types']}")
    print(f"\nLabel Distribution:")
    for label, count in sorted(stats['statistics']['label_distribution'].items()):
        print(f"  - {label}: {count} examples")
    
    # 5. Test predictions on similar clauses
    print_section("Step 5: Testing on Similar Clauses")
    
    test_clauses = [
        "The parties shall maintain insurance coverage as required by law.",
        "Customer may inspect the books and records upon reasonable notice.",
        "This agreement renews automatically unless terminated 60 days prior."
    ]
    
    for clause in test_clauses:
        print(f"\nTest: {clause}")
        result = predict_text(clause)
        
        if result['status'] == 'success' and result['result']:
            top_prediction = result['result'][0]
            confidence = top_prediction.get('confidence', 0)
            confidence_indicator = "HIGH" if confidence >= 0.75 else "MEDIUM" if confidence >= 0.55 else "LOW"
            
            print(f"  [{confidence_indicator}] Predicted: {top_prediction['clause_type']}")
            print(f"     Confidence: {confidence*100:.1f}%")
            print(f"     Source: {top_prediction.get('source', 'unknown')}")
    
    # 6. Summary
    print_section("Summary")
    print("Model learned new clause types from user feedback")
    print("Knowledge persisted to support_set.pkl")
    print("Future similar clauses will be classified correctly")
    print("No retraining required - immediate adaptation")
    print("\nThis demonstrates the power of meta-learning for legal AI.")
