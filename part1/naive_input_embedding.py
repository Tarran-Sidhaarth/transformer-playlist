import random

# 1. Define our dimensions
vocab_size = 256
embed_dim = 4
context_length = 6

# 2. Create the Embedding Table (Matrix)
# We create a list of 256 rows. Each row is a list of 4 random decimal numbers.
# In a real model, these start random and are adjusted during training.
embedding_table = []
for _ in range(vocab_size):
    # Create a vector of 4 random numbers between -1 and 1
    vector = [round(random.uniform(-1.0, 1.0), 4) for _ in range(embed_dim)]
    embedding_table.append(vector)

# Print the shape of our table
print(f"Embedding Table Shape: {len(embedding_table)} rows x {len(embedding_table[0])} columns")

# 3. Our Input Sequence
# The word "butter" in raw bytes
input_tokens = [98, 117, 116, 116, 101, 114]
print(f"Input Shape: ({len(input_tokens)},) -> {input_tokens}\n")

# 4. The Forward Pass (Getting the Input Embeddings)
input_embeddings = []
for token_id in input_tokens:
    # Look up the row in the table corresponding to the token ID
    vector = embedding_table[token_id]
    input_embeddings.append(vector)

# 5. The Output
print("--- Resulting Input Embeddings ---")
for i, token_id in enumerate(input_tokens):
    print(f"Token {token_id:3} -> {input_embeddings[i]}")

print(f"\nOutput Shape: ({len(input_embeddings)}, {len(input_embeddings[0])})")