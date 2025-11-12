from enum import Enum

# ==============================================================================
# Optimizador: declaracion de optimizador como ENUM
# ==============================================================================
# https://docs.pytorch.org/docs/stable/optim.html


class Optimizer(Enum):
    SGD = "SGD"
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSProp"
