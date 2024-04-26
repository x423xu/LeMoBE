
import torch
import torch.nn as nn
from NVAE.model import AutoEncoder

class ShapeNetAutoEncoder(AutoEncoder):
    def __init__(self, args, writer, arch_instance):
        super().__init__(args, writer, arch_instance)