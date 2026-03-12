import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import os
from mesa_pipeline import MESAPipeline

def train_mesa():
    print("[INFO] Initializing MESA Dual-Objective Training...")
    mesa = MESAPipeline()
    
    # Load SQuAD dataset for Meta-Learning
    dataset = load_dataset("squad", split="train").shuffle(seed=42).select(range(5000)) 
    
    # Text corpus used for KL-Divergence Generative Regularization
    reg_text = "The quick brown fox jumps over the lazy dog. Artificial intelligence is a fascinating field of computer science."
    reg_inputs = mesa.tokenizer(reg_text, return_tensors="pt").to(mesa.device)
    
    optimizer = torch.optim.AdamW(mesa.hypernet.parameters(), lr=1e-4)
    beta = 0.5 # KL Divergence Penalty Weight
    accumulation_steps = 4 
    epochs = 2
    
    os.makedirs("checkpoints", exist_ok=True)
    mesa.hypernet.train()

    print(f"[INFO] Beginning Training Run (Epochs: {epochs}, Batches: {len(dataset)})")
    
    for epoch in range(epochs):
        total_ce, total_kl = 0, 0
        pbar = tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        
        for step, data in enumerate(pbar):
            context = f"Fact: {data['context']}"
            answer = data['answers']['text'][0] if len(data['answers']['text']) > 0 else "Unknown."
            
            # Embed & Generate
            doc_emb = mesa.get_document_embedding(context)
            Wa, Wb = mesa.hypernet(doc_emb)
            
            # --- Objective 1: Generative Stability (KL Divergence) ---
            with torch.no_grad():
                base_logits = mesa.model(**reg_inputs).logits

            mesa.inject(Wa, Wb, scaling_factor=2.0)
            eph_logits = mesa.model(**reg_inputs).logits

            kl_loss = F.kl_div(
                F.log_softmax(eph_logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction='batchmean'
            )

            # --- Objective 2: Factual Plasticity (Cross Entropy) ---
            messages =[{"role": "system", "content": "You are a factual assistant. Be brief."},
                        {"role": "user", "content": data['question']}]
            prompt = mesa.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            prompt_ids = mesa.tokenizer(prompt, return_tensors="pt").input_ids[0]
            answer_ids = mesa.tokenizer(answer + mesa.tokenizer.eos_token, return_tensors="pt").input_ids[0]
            
            input_ids = torch.cat([prompt_ids, answer_ids]).unsqueeze(0).to(mesa.device)
            labels = input_ids.clone()
            labels[0, :len(prompt_ids)] = -100 # Mask prompt to isolate generation
            
            ce_loss = mesa.model(input_ids=input_ids, labels=labels).loss
            
            # Optimization
            loss = (ce_loss + (beta * kl_loss)) / accumulation_steps
            loss.backward()
            mesa.cleanup()
            
            total_ce += ce_loss.item()
            total_kl += kl_loss.item()
            
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(mesa.hypernet.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix({"CE Loss": f"{total_ce / (step + 1):.3f}", "KL Div": f"{total_kl / (step + 1):.4f}"})
        
        torch.save(mesa.hypernet.state_dict(), f"checkpoints/mesa_kl_ep{epoch+1}.pt")
        print(f"[INFO] Saved Checkpoint: checkpoints/mesa_kl_ep{epoch+1}.pt")

if __name__ == "__main__":
    train_mesa()