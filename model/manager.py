import os
import torch
from model.abcmodel import CONFIG_OPERATION

class ModelManager:
    """
    To manage saving, loading, and exporting torch models.
    """
    def __init__(self, model:torch.nn.Module):
        self.model = model
        self.cfg_file = "cfg.ckpt"
        self.model_file = "weights.ckpt"
        self.optimizer_file = "optimizer.ckpt"

    def dump(self, ckpt:str,
            num_epochs=0,
            global_step=0,
            dump_optimizer=False):
        os.makedirs(ckpt, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict()
            }, f"{ckpt}/{self.model_file}"
        )
        torch.save({
            "model_cfg": CONFIG_OPERATION(self.model.cfg)
            ,"num_epochs": num_epochs
            ,"global_step": global_step
            }, f"{ckpt}/{self.cfg_file}"
        )
        if dump_optimizer:
            torch.save({
                "optimizer_state_dict": self.model.optimizer.state_dict()
                }, f"{ckpt}/{self.optimizer_file}"
            )

        print(f"Dumped checkpoint {ckpt} successfully.", flush=True)
            
    def load(self, ckpt:str, 
            load_optimizer=False,
            rank=0):
        files = [f"{ckpt}/{self.cfg_file}", f"{ckpt}/{self.model_file}"]
        if load_optimizer:
            files.append(f"{ckpt}/{self.optimizer_file}")
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Load model error: no such file:{file}")

        checkpoint = torch.load(f"{ckpt}/{self.cfg_file}", weights_only=False, map_location="cpu")
        if CONFIG_OPERATION(self.model.cfg) != CONFIG_OPERATION(checkpoint["model_cfg"]):
            self.model.cfg.update(CONFIG_OPERATION(checkpoint["model_cfg"]))
            self.model.init(self.model.cfg) #reinitailize model
        num_epochs = checkpoint["num_epochs"] 
        global_step = checkpoint["global_step"]

        device = self.model.device
        checkpoint = torch.load(f"{ckpt}/{self.model_file}", weights_only=False, map_location="cpu")
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in checkpoint["model_state_dict"]:
                    param.copy_(checkpoint["model_state_dict"][name].to(param.dtype)).to(device)
                else:
                    print(f"Rank {rank} Warning: {name} not found in state_dict.", flush=True)
        
        if load_optimizer:
            checkpoint = torch.load(f"{ckpt}/{self.optimizer_file}", weights_only=False, map_location="cpu")
            self.model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Rank {rank} Loaded checkpoint {ckpt} with {num_epochs} epochs and step {global_step}.", flush=True)

        return num_epochs, global_step


