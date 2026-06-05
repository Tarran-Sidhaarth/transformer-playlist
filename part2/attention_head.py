import torch
import torch.nn as nn
import math

generator = torch.Generator()
generator.manual_seed(42)


CONTEXT_LENGTH = 4
EMBEDDING_DIMENSION = 6

X = torch.randn(CONTEXT_LENGTH,EMBEDDING_DIMENSION, generator=generator)

WQ = nn.Linear(EMBEDDING_DIMENSION,EMBEDDING_DIMENSION, bias=False)
WK = nn.Linear(EMBEDDING_DIMENSION,EMBEDDING_DIMENSION, bias=False)
WV = nn.Linear(EMBEDDING_DIMENSION,EMBEDDING_DIMENSION, bias=False)

Q = WQ(X) # X @ WQ
K = WK(X)
V = WV(X)

print(f"Q,K,V shapes: {Q.shape}, {K.shape}, {V.shape}")

# K(context_length, embedding_dimensions) -> K(embedding_dimension, context_length)

similarity = (Q @ K.transpose(0,1))/math.sqrt(EMBEDDING_DIMENSION)
normalized_score = similarity.softmax(-1)

output = normalized_score @ V

print(normalized_score)
print(output.shape)

