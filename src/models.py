import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)
        
        # reshape logits because PyTorch expects targets to be (B, T)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)  # or use targets.view(-1)

        # negative log-likelihood loss
        loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx: torch.tensor, max_new_tokens: int) -> torch.tensor:
        """Generate predictions for new tokens.

        Args:
            idx (torch.tensor): (B, T) array of indices in the current context.
            max_new_tokens (int): _description_
        """
        
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
            
        return idx