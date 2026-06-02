import torch
import math

D_MODEL = 4
MAX_POSITIONS = 10

def PE(pos: int, d_model: int) -> torch.Tensor:
    """
    Returns the  positional encoding vector for a given position.

    Args:
        pos:     Token position index (0-based).
        d_model: Dimensionality of the model embeddings (must be even).

    Returns:
        A 1D tensor of shape (d_model,).
    """

    pe = torch.zeros(d_model,)

    for i in range(d_model//2):
        denom = math.pow(1e4,(2*i/d_model))
        pe[2*i] = math.sin(pos/denom)
        pe[2*i+1] = math.cos(pos/denom)

    return pe

def get_all_PE(max_positions: int = MAX_POSITIONS, d_model: int = D_MODEL,) -> torch.Tensor:
    """
    Returns positional encodings for all positions stacked into a matrix.

    Returns:
        Tensor of shape (max_positions, d_model).
    """
    return torch.stack(PE(pos) for pos in range(max_positions)) #type: ignore

def print_positional_encodings(
    max_positions: int = MAX_POSITIONS,
    d_model: int = D_MODEL,
) -> None:
    """Prints a formatted table of all positional vectors."""
    header = "  ".join(f"{'dim_'+str(d):>9}" for d in range(d_model))
    print(f"Positional Encodings | d_model={d_model}\n")
    print(f"{'pos':>4}  {header}")
    print("-" * (4 + 11 * d_model))

    for pos in range(max_positions):
        vec = PE(pos, d_model)
        vals = "  ".join(f"{v.item():>9.5f}" for v in vec)
        print(f"{pos:>4}  {vals}")

print_positional_encodings()
