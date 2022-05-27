from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel
import torch.nn as nn
import torch.functional as F

class ADUCls(nn.Module):
    def __init__(self) -> None:
        super().__init__()