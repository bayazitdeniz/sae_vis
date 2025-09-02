import re
from dataclasses import dataclass
from typing import Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
import einops

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class CrossCoder(nn.Module):
    def __init__(self, cfg, is_btopk=False):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        n_models = 2 # default is to have at least 2 models
        if "n_models" in self.cfg.keys():
            n_models = self.cfg["n_models"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.empty(n_models, d_in, d_hidden, dtype=self.dtype)
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(
                    d_hidden, n_models, d_in, dtype=self.dtype
                )
            )
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((n_models, d_in), dtype=self.dtype)
        )
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0
        
        self.is_btopk = is_btopk
        if is_btopk:
            self.register_buffer("k", torch.tensor(cfg["batch_topk_init"], dtype=torch.int))
            threshold = -1.0
            self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            f = F.relu(x_enc + self.b_enc)
        else:
            f = x_enc + self.b_enc
        
        if self.is_btopk:
            post_relu_f = f
            code_normalization = self.W_dec.norm(dim=2).sum(dim=1).unsqueeze(0)
            post_relu_f_scaled = post_relu_f * code_normalization
            f = post_relu_f * (post_relu_f_scaled > self.threshold)
        
        return f

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)
    
    @classmethod
    def load(cls, version_dir, checkpoint_version, path="./workspace/logs/checkpoints", verbose=True):
        # TODO: fix base dir naming to be model agnostic
        #       for now keep it this way because 
        #       the path is hardcoded in the analysis
        save_dir = Path(path) / str(version_dir)
        cfg_path = save_dir / f"{str(checkpoint_version)}_cfg.json"
        weight_path = save_dir / f"{str(checkpoint_version)}.pt"

        cfg = json.load(open(cfg_path, "r"))
        if verbose:
            pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(weight_path))
        return self

# # ==============================================================
# # ! TRANSFORMERS
# # This returns the activations & resid_pre as well (optionally)
# # ==============================================================

from nnsight import LanguageModel
class TransformerLensWrapper(nn.Module):
    """
    This class wraps around & extends the TransformerLens model, so that we can make sure things like the forward
    function have a standardized signature.
    """

    def __init__(self, model: LanguageModel, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.hook_point = "resid"
        LAYER = int(self.cfg.hook_point.split(".")[1]) - 1
        model_config = model.config
        model_name_lowered = model_config._name_or_path.lower()
        if hasattr(model_config, "_name_or_path"):
            if "pythia" in model_name_lowered:
                self.submodule = model.gpt_neox.layers[LAYER]
            elif "olmo" in model_name_lowered:
                self.submodule = model.model.layers[LAYER]
            elif "bloom" in model_name_lowered:
                self.submodule = model.transformer.h[LAYER]
            else:
                raise NotImplementedError("Model name not supported yet.")
        else:
            raise ValueError("Model config is missing _name_or_path entry.")

    @overload
    def forward(
        self,
        tokens: Tensor,
        return_logits: Literal[True],
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        tokens: Tensor,
        return_logits: Literal[False],
    ) -> Tensor: ...

    def forward(
        self,
        tokens: Int[Tensor, "batch seq"],
        return_logits: bool = True,
    ):
        """
        Inputs:
            tokens: Int[Tensor, "batch seq"]
                The input tokens, shape (batch, seq)
            return_logits: bool
                If True, returns (logits, residual, activation)
                If False, returns (residual, activation)
        """
        # Store the activations & final value of residual stream
        # If return_logits is False, then we compute the last residual stream value but not the logits
        
        with self.model.trace(tokens, **{'scan' : False, 'validate' : False}): #, invoker_args={}): 
            hidden_states = self.submodule.output.save()
            
            # Capture hidden states (activations)
            hidden_states = self.submodule.output.save()
            curr_input = self.model.inputs.save()
            if "pythia" in self.model.config._name_or_path.lower():
                resid_pre = self.model.gpt_neox.layers[-1].output[0].save()
            elif "olmo" in self.model.config._name_or_path.lower():
                resid_pre = self.model.model.layers[-1].output[0].save()
            elif "bloom" in self.model.config._name_or_path.lower():
                resid_pre = self.model.transformer.h[-1].output[0].save()
            else:
                raise NotImplementedError("Model name not supported yet.")

            # Stop capturing after saving
            self.submodule.output.stop()
       
        attn_mask = curr_input.value[1]["attention_mask"]
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = hidden_states[attn_mask != 0]
        hidden_states = hidden_states.view(tokens.shape[0], tokens.shape[1], hidden_states.shape[-1])

        # The hook functions work by storing data in model's hook context, so we pop them back out
        activation: Tensor = hidden_states
        return resid_pre, activation

    def hook_fn_store_act(self, activation: torch.Tensor, hook: HookPoint):
        hook.ctx["activation"] = activation

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def W_U(self):
        # return self.model.W_U
        if hasattr(self.model, "embed_out"):
            return self.model.embed_out.weight.T
        else:
            return self.model.lm_head.weight.T

    @property
    def W_out(self):
        return self.model.W_out


def to_resid_dir(dir: Float[Tensor, "feats d_in"], model: TransformerLensWrapper):
    """
    Takes a direction (eg. in the post-ReLU MLP activations) and returns the corresponding dir in the residual stream.

    Args:
        dir:
            The direction in the activations, i.e. shape (feats, d_in) where d_in could be d_model, d_mlp, etc.
        model:
            The model, which should be a HookedTransformerWrapper or similar.
    """
    # If this SAE was trained on the residual stream or attn/mlp out, then we don't need to do anything
    if "resid" in model.hook_point or "_out" in model.hook_point:
        return dir

    # If it was trained on the MLP layer, then we apply the W_out map
    elif ("pre" in model.hook_point) or ("post" in model.hook_point):
        return dir @ model.W_out[model.hook_layer]

    # Others not yet supported
    else:
        raise NotImplementedError(
            "The hook your SAE was trained on isn't yet supported"
        )
