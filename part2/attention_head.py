import torch
import torch.nn as nn
import math

generator = torch.Generator()
generator.manual_seed(42)


CONTEXT_LENGTH = 4
EMBEDDING_DIMENSION = 6

