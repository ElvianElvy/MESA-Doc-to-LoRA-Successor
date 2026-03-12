import torch
import time
from mesa_pipeline import MESAPipeline
import warnings

warnings.filterwarnings("ignore")

def evaluate_mesa():
    print("[INFO] Loading MESA Pipeline...")
    mesa = MESAPipeline()
    
    checkpoint_path = "checkpoints/mesa_kl_ep2.pt"
    mesa.hypernet.load_state_dict(torch.load(checkpoint_path, map_location=mesa.device))
    mesa.hypernet.eval()
    print(f"[INFO] Successfully loaded Hypernetwork weights from {checkpoint_path}")

    test_cases =[
        {
            "id": "Medical OOD",
            "context": "Fact: The rare disease 'Crimson-Scale' is cured by applying crushed bioluminescent mushrooms from the Marianas Trench.",
            "question": "What is the cure for Crimson-Scale?"
        },
        {
            "id": "Corporate OOD",
            "context": "Fact: CyberDyne's Q3 revenue of $45.2 billion was driven by the Maid-O-Matic 9000 cleaning robot.",
            "question": "What drove CyberDyne Systems' revenue in Q3?"
        }
    ]

    print("\n" + "="*50)
    print("MESA ZERO-CONTEXT INFERENCE EVALUATION")
    print("="*50)
    
    for test in test_cases:
        print(f"\n[Test Case]: {test['id']}")
        print(f"Context   : {test['context']}")
        print(f"Question  : {test['question']}")
        
        t0 = time.time()
        
        # System 2: Generate Weights
        doc_emb = mesa.get_document_embedding(test['context'])
        with torch.no_grad():
            Wa, Wb = mesa.hypernet(doc_emb)
        
        # System 3: Inject and Generate
        mesa.inject(Wa, Wb, scaling_factor=2.0)
        
        try:
            messages =[{"role": "system", "content": "You are a factual assistant. Answer accurately."},
                        {"role": "user", "content": test['question']}]
            prompt = mesa.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = mesa.tokenizer([prompt], return_tensors="pt").to(mesa.device)
            
            with torch.no_grad():
                outputs = mesa.model.generate(**inputs, max_new_tokens=40, temperature=0.1, pad_token_id=mesa.tokenizer.eos_token_id)
            
            response = mesa.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            t1 = time.time()
            
            print(f"MESA Output: {response}")
            print(f"Latency    : {(t1-t0):.3f}s")
            
        finally:
            mesa.cleanup()

if __name__ == "__main__":
    evaluate_mesa()