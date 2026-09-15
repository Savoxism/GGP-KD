"""Numerical and runtime constants shared by the GGPKD graph, collate and loss.

None of these defines the objective: changing one changes precision, memory or
speed, never what is optimized.
"""

EPS_NORM = 1e-8

# How a step's pool is cut into student forward calls. With correct attention
# masks neither value can change an embedding; they only trade padded FLOPs
# against kernel launches, and the right trade is GPU-specific.
ENCODE_CHUNK_SIZE = 256
PAD_TO_MULTIPLE_OF = 8
