import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

class DeepWeightProgrammer(nn.Module):
    """System 2: Generates LoRA weights from Context Embeddings."""
    def __init__(self, in_dim=896, out_dim=128, lora_rank=16, num_targets=3): 
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.lora_rank = lora_rank
        self.num_targets = num_targets 
        
        self.compressor = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU()
        )
        self.generate_A = nn.Linear(512, num_targets * lora_rank * in_dim)
        self.generate_B = nn.Linear(512, num_targets * out_dim * lora_rank)
        
        # Initialize B to 0 to prevent initial catastrophic forgetting
        nn.init.zeros_(self.generate_B.weight)
        nn.init.zeros_(self.generate_B.bias)

    def forward(self, doc_embedding):
        x = self.compressor(doc_embedding)
        A_flat = self.generate_A(x)
        B_flat = self.generate_B(x)
        Wa = A_flat.view(self.num_targets, self.lora_rank, self.in_dim)  
        Wb = B_flat.view(self.num_targets, self.out_dim, self.lora_rank) 
        return Wa, Wb

class MESAPipeline:
    """Master Pipeline for Macular Ephemeral State-space Architecture."""
    def __init__(self, base_model="Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.model.eval() # Base model remains completely frozen

        # System 3 Target Layers (v_proj)
        self.target_layers =[13, 14, 15]
        self.target_modules = [self.model.model.layers[l].self_attn.v_proj for l in self.target_layers]
        
        # Cache original base model states
        self.original_forwards = [m.forward for m in self.target_modules]
        self.original_weights =[m.weight.clone().detach() for m in self.target_modules]
        self.original_biases =[(m.bias.clone().detach() if m.bias is not None else None) for m in self.target_modules]

        self.hypernet = DeepWeightProgrammer(num_targets=len(self.target_layers)).to(self.device, dtype=torch.bfloat16)

    def get_document_embedding(self, text):
        """Extracts LLM contextual embedding for the Hypernetwork."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)

    def inject(self, Wa, Wb, scaling_factor=2.0):
        """System 3: Dynamically modifies the attention Value projection."""
        self.delta_ws =[]
        for i, module in enumerate(self.target_modules):
            delta_w = (Wb[i] @ Wa[i]) * scaling_factor
            self.delta_ws.append(delta_w)
            
            def make_dynamic_forward(orig_w, orig_b, dw):
                def dynamic_forward(x):
                    return F.linear(x, orig_w + dw, orig_b)
                return dynamic_forward
                
            module.forward = make_dynamic_forward(self.original_weights[i], self.original_biases[i], delta_w)

    def cleanup(self):
        """Restores Base LLM state and frees KV-cache overhead."""
        for i, module in enumerate(self.target_modules):
            module.forward = self.original_forwards[i]