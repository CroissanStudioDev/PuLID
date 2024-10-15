import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from safetensors.torch import load_file

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.relu(self.down(x))) * self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

def create_network_from_weights(state_dict, multiplier=1.0):
    network = {}
    for key, value in state_dict.items():
        if 'lora_down' in key:
            up_key = key.replace('lora_down', 'lora_up')
            down_weight = value
            up_weight = state_dict[up_key]
            
            layer_name = '.'.join(key.split('.')[:-2])
            rank = down_weight.shape[0]
            
            alpha = 1.0
            if f'{layer_name}.alpha' in state_dict:
                alpha = state_dict[f'{layer_name}.alpha'].item()
            
            network[layer_name] = (down_weight, up_weight, alpha * multiplier)
    
    return network

def load_lora_weights(model, lora_path, multiplier=1.0):
    if lora_path.endswith('.safetensors'):
        lora_state_dict = load_file(lora_path)
    else:
        lora_state_dict = torch.load(lora_path, map_location='cpu')
    
    network = create_network_from_weights(lora_state_dict, multiplier)
    
    for name, module in model.named_modules():
        if name in network:
            down_weight, up_weight, alpha = network[name]
            
            if isinstance(module, LoRALayer):
                module.down.weight.data.copy_(down_weight)
                module.up.weight.data.copy_(up_weight)
                module.alpha = alpha
            elif isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features, down_weight.shape[0], alpha)
                lora.down.weight.data.copy_(down_weight)
                lora.up.weight.data.copy_(up_weight)
                
                new_module = nn.Sequential(module, lora)
                setattr(model, name, new_module)
    
    return model

def add_lora_to_model(model, target_modules=["qkv", "proj", "fc1", "fc2"], rank=4, alpha=1):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features, rank, alpha)
                new_module = nn.Sequential(module, lora)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, new_module)
    return model
