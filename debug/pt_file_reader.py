import torch
import os

def load_checkpoint(pytorch_model_path):
    model = torch.load(pytorch_model_path)
    state_dict = model["model"]
    for key in state_dict.keys():
        value = state_dict[key]
        if "layers.0" in key and not "min" in key and not "max" in key and ("scaling" in key or "integer" in key):
            print(key, value.shape)
        #if "scaling" in key or "min" in key or "max" in key:
        #    print(key, value)
        #if "integer" in key:
        #    print(state_dict[key])
        #    break
    return state_dict
model_dir = "/home/user/shared_docker/I-BERT/outputs/symmetric/QQP-base/wd0.1_ad0.1_d0.1_lr1e-06/0919-202918_ckpt"
model_name = "checkpoint_best.pt"
model_path = os.path.join(model_dir, model_name)
load_checkpoint(model_path)