import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.embed_dim = config.embed_dim
        
        # Define query, key, value projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project inputs to q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Output head
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        
        # Get token and position embeddings
        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_head(x)
        
        return logits

class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 vocabulary size
        max_seq_len=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        dropout=0.1
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout = dropout

# Training utilities
def create_causal_mask(seq_len):
    """Creates a causal mask for self-attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask

class TrainingArgs:
    def __init__(
        self,
        learning_rate=3e-4,
        batch_size=32,
        num_epochs=10,
        warmup_steps=1000,
        max_grad_norm=1.0,
        save_steps=1000,
        eval_steps=1000
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps

def train_step(model, optimizer, batch, device):
    """Performs a single training step."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # Create causal mask
    mask = create_causal_mask(input_ids.size(1)).to(device)
    
    # Forward pass
    logits = model(input_ids, mask)
    
    # Compute loss (shift logits and labels for next-token prediction)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Backward pass
    loss.backward()
    
    return loss.item()

# Example usage:
if __name__ == "__main__":
    # Initialize configuration
    config = GPTConfig(
        vocab_size=50257,
        max_seq_len=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12
    )
    
    # Initialize model
    model = GPT(config)
    
    # Initialize training arguments
    training_args = TrainingArgs(
        learning_rate=3e-4,
        batch_size=32,
        num_epochs=10
    )
    
    # Example of creating a dummy batch
    batch = {
        "input_ids": torch.randint(0, config.vocab_size, (training_args.batch_size, config.max_seq_len)),
        "labels": torch.randint(0, config.vocab_size, (training_args.batch_size, config.max_seq_len))
    }
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training step
    loss = train_step(model, optimizer, batch, device)
    print(f"Training loss: {loss}")