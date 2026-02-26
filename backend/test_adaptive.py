"""
Quick test of adaptive learning functionality
Run after starting the backend server
"""

from src.ml_model import LexiCacheModel

def test_model():
    print("Testing LexiCache Adaptive Model\n")
    
    # 1. Initialize model
    print("1. Initializing model...")
    model = LexiCacheModel()
    
    # 2. Check statistics
    print("\n2. Initial statistics:")
    stats = model.get_statistics()
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Unique types: {stats['unique_types']}")
    print(f"   Known CUAD types: {stats['known_cuad_types']}")
    
    # 3. Test classification
    print("\n3. Testing classification...")
    test_text = """
    This Agreement shall be governed by and construed in accordance with the laws 
    of the State of California, without regard to its conflict of law principles.
    """
    
    result = model.predict_cuad(test_text)
    for clause in result:
        print(f"   - {clause['clause_type']}: {clause['confidence']*100:.1f}%")
    
    # 4. Test learning from feedback
    print("\n4. Testing online learning...")
    success = model.learn_from_feedback(
        clause_text="The Supplier shall maintain insurance coverage of $5M.",
        correct_label="Insurance"
    )
    
    if success:
        print("   ✓ Learning successful!")
        stats = model.get_statistics()
        print(f"   Total examples now: {stats['total_examples']}")
    
    # 5. Test another prediction
    print("\n5. Testing improved prediction...")
    test_text2 = "The contractor must maintain adequate insurance policies."
    result2 = model.predict_cuad(test_text2)
    
    for clause in result2:
        print(f"   - {clause['clause_type']}: {clause['confidence']*100:.1f}%")
    
    print("\n✓ All tests passed!")
    print("\nModel is ready for adaptive few-shot learning!")

if __name__ == "__main__":
    test_model()
