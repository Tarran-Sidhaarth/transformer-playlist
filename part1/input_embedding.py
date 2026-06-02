import torch
import torch.nn as nn

vocab_size = 256
embed_dim = 4

embedding_table = nn.Embedding(vocab_size,embed_dim)

input_tokens = torch.tensor([98, 117, 116, 116, 101, 114])

input_embeddings = embedding_table(input_tokens)

print("--- PyTorch Results ---")
print(f"Input Shape:  {list(input_tokens.shape)}") #type: ignore
print(f"Output Shape: {list(input_embeddings.shape)}")
print("\nOutput Tensor:")
print(input_embeddings)