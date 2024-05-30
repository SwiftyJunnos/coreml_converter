from typing import Optional

import torch
import torch.nn as nn


def load_weight(
    model: nn.Module, weight_path: str, device: torch.device
) -> Optional[nn.Module]:
    # .bin
    if weight_path.lower().endswith(".bin"):
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        return model
    return None
