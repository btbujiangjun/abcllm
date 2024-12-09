#
# download & load gpt2 pretrain model
# include:
#   small, medium, large, xl
#

import os
import json
import urllib
import torch
import numpy as np
from tqdm import tqdm
import tensorflow as tf 
from model.model import GPTModel

class PretrainGPT2:
    def __init__(self):
        self.BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,
            "qkv_bias": True,
            "lr": 5e-5,
            "decay": 0.1,
            "device": torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        }
        self.MODEL_CONFIG ={
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

    def load_tf_ckpt(self, choose_model, ckpt_dir, dtype=torch.bfloat16):
        
        model, model_size = self.init_model(choose_model)
        
        params = self.__load_params(ckpt_dir, model_size)
        model.pos_emb.weight = self.__assign(model.pos_emb.weight, params['wpe'])
        model.tok_emb.weight = self.__assign(model.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            model.trf_blocks[b].att.w_query.weight = self.__assign(
                model.trf_blocks[b].att.w_query.weight, q_w.T)
            model.trf_blocks[b].att.w_key.weight = self.__assign(
                model.trf_blocks[b].att.w_key.weight, k_w.T)
            model.trf_blocks[b].att.w_value.weight = self.__assign(
                model.trf_blocks[b].att.w_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            model.trf_blocks[b].att.w_query.bias = self.__assign(
                model.trf_blocks[b].att.w_query.bias, q_b)
            model.trf_blocks[b].att.w_key.bias = self.__assign(
                model.trf_blocks[b].att.w_key.bias, k_b)
            model.trf_blocks[b].att.w_value.bias = self.__assign(
                model.trf_blocks[b].att.w_value.bias, v_b)

            model.trf_blocks[b].att.out_proj.weight = self.__assign(
                model.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            model.trf_blocks[b].att.out_proj.bias = self.__assign(
                model.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])

            model.trf_blocks[b].ff.layers[0].weight = self.__assign(
                model.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            model.trf_blocks[b].ff.layers[0].bias = self.__assign(
                model.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            model.trf_blocks[b].ff.layers[2].weight = self.__assign(
                model.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            model.trf_blocks[b].ff.layers[2].bias = self.__assign(
                model.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            model.trf_blocks[b].norm1.scale = self.__assign(
                model.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            model.trf_blocks[b].norm1.shift = self.__assign(
                model.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            model.trf_blocks[b].norm2.scale = self.__assign(
                model.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            model.trf_blocks[b].norm2.shift = self.__assign(
                model.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])

        model.final_norm.scale = self.__assign(model.final_norm.scale, params["g"])
        model.final_norm.shift = self.__assign(model.final_norm.shift, params["b"])
        model.out_head.weight = self.__assign(model.out_head.weight, params["wte"])
        model.to(dtype).to(self.BASE_CONFIG["device"])
        model.reset_optimizer()

        print(f"loading gpt2 tf_ckpt finished.")

        return model

    def init_model(self, choose_model, dtype=torch.bfloat16):
        assert choose_model in self.MODEL_CONFIG.keys(), (
            f"model {choose_model} not exist. plz choose :{self.MODEL_CONFIG.keys}"
        )

        self.BASE_CONFIG.update(self.MODEL_CONFIG[choose_model])
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        model = GPTModel(self.BASE_CONFIG).to(dtype)
        
        return model, model_size
    
    def __load_params(self, ckpt_dir, model_size):
        allowed_sizes = ("124M", "355M", "774M", "1558M")
        if model_size not in allowed_sizes:
            raise ValueError(f"Model size not in {allowed_sizes}")

        ckpt_dir = os.path.join(ckpt_dir, model_size)
        base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        files = [
            "checkpoint", "encoder.json", "hparams.json",
            "model.ckpt.data-00000-of-00001", "model.ckpt.index",
            "model.ckpt.meta", "vocab.bpe"
        ]

        os.makedirs(ckpt_dir, exist_ok=True)
        for file in files:
            file_url = os.path.join(base_url, model_size, file)
            file_path = os.path.join(ckpt_dir, file)
            self.__download(file_url, file_path)

        tf_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        settings = json.load(open(os.path.join(ckpt_dir, "hparams.json")))
        params = self.__load_tf_ckpt(tf_ckpt_path, settings)

        return params

    def __download(self, url, destination):
        try:
            with urllib.request.urlopen(url) as response:
                file_size = int(response.headers.get("Content-Length", 0))

                if os.path.exists(destination):
                    file_size_local = os.path.getsize(destination)
                    if file_size_local == file_size:
                        print(f"File already exists and is up-to-date:{destination}")
                        return

                block_size = 1024 
                progress_bar_description = os.path.basename(url)  # Extract filename from URL
                with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                    with open(destination, "wb") as file:
                        while True:
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                            file.write(chunk)
                            progress_bar.update(len(chunk))  # Update progress bar
        except urllib.error.HTTPError:
            s = (
                f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
                "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
                " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
            print(s)

    def __load_tf_ckpt(self, ckpt_path, settings):
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}
        
        for name, _ in tf.train.list_variables(ckpt_path):
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
            variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]

            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})

            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array

        return params

    def __assign(self, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

        return torch.nn.Parameter(torch.tensor(right).to(left.device))






