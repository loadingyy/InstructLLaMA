#  Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""

import math
from typing import Dict, List

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


import instruct_llama.model as llama


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        """Store LoRA specific attributes in a class.

        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            merge_weights: whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
                fine-tuned model as a standalone one (without storing LoRA weights separately) plus it helps to reduce
                overhead during inference.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, num_embeddings: int, embedding_dim: int, r: int = 0, lora_alpha: int = 1, merge_weights: bool = True, **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x,
                self.lora_A.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.weight` (because of the nn.Linear inheritance)
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA for all three (query, key and value) we can set it as False. For example if we want
                to apply LoRA only to `query` and `value` but keep `key` without weight updates we should pass `[True,
                False, True]`
            fan_in_fan_out: set this to True if the layer to replace stores weight like (fan_in, fan_out).  For example, gpt-2 uses
                `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`
                https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#LL53C9-L53C112
            merge_weights: whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
                fine-tuned model as a standalone one (without storing LoRA weight separately) plus it helps to reduce
                overhead during inference.
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))  # (4, 128)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))  # (256, 2)
            )  # weights for Conv1D with groups=sum(enable_lora)
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False  # (384, 128)

            # Compute the indices
            # Indices are needed to properly pad weight updates with zeros. If we want to fine-tune queries and values,
            # but not keys, then the weights update should be:
            #
            # [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
            #  [....................................],
            #  [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            #      ↑              ↑            ↑
            # ________________________________________
            # | query         | key       | value    |
            # ----------------------------------------
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)  # (3, 128)
            self.lora_ind[enable_lora, :] = True  # (3, 128)
            self.lora_ind = self.lora_ind.view(-1)  # (384,)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)
        # Note: double transpose (in the beginning and in the end) is basically a guard for two-dimensional tensors
        # for example when we want to merge/unmerge LoRA weights and pretrained weights
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.out_features))  # (64, 64, 384)
        result = result.view(-1, self.out_features)  # (4096, 384)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )  # (4096, 256)
        return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)  # (64, 64, 384)

    def train(self, mode: bool = True):
        """Set the module into train or eval mode if `mode` is True of False respectively.

        For train mode (train(True)) if weights are merged we need to subtract weights updates (LoRA_A @ LoRA_B) from
        pretrained weights so we can continue training LoRA's matrices A and B and keep pretrained weights frozen.

        For eval mode (train(False)) if weights are not merged we need to add weight updates to pretrained weights in
        order to reduce computational overhead during inference.

        Args:
            mode: if True the module will be set into train mode (affects Dropout and BatchNorm), if False - eval mode.

        """

        def T(w):
            return w.T if self.fan_in_fan_out else w

        # despite being called from nn.Linear this method will put all layers into train mode, including nn.Dropout
        # of course except parameters (such as self.lora_A, self.lora_B)
        nn.Linear.train(self, mode)

        # if train(True) -> unmerge unless we already have them unmerged
        # if train(False) -> merge unless we already have them merged
        should = self.merged if mode else not self.merged

        # Let's assume that:
        # ⚬ self.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        if self.merge_weights and should:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                    self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
                    groups=sum(self.enable_lora),
                ).squeeze(
                    0
                )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
                # -1: W = W - delta_W (unmerge), +1: W = W + delta_W (merge)
                sign = -1 if mode else 1
                self.weight.data += sign * self.zero_pad(T(delta_w * self.scaling))  # (256, 128) after zero_pad (384, 128)
            self.merged = not mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        def T(w):
            return w.T if self.fan_in_fan_out else w

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # the logic here is that the weights are merged only during inference
        # so if they are merged we don't need to do anything with LoRA's A and B matrices
        # but if the weights are not merged that means that the forward method is called during
        # training and we need to forward pass input through pretrained weights, LoRA A and B matrices
        # and do the summation (as per scheme at the top of the file)
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            # `F.linear` automatically transposes the second argument (T(self.weight) in our case)
            result = F.linear(x, T(self.weight), bias=self.bias)  # (64, 64, 128) @ (384, 128) -> (64, 64, 384)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
                # For F.conv1d:
                # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
                # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
                # ⚬ groups: split input into groups, in_channels should be divisible by the number of groups. Default: 1
                # presumably iW - sequence width/length, kW - kernel width
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
                    self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
                    groups=sum(self.enable_lora),
                ).transpose(
                    -2, -1
                )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
                result += self.zero_pad(after_B) * self.scaling  # (64, 64, 256) after zero_pad (64, 64, 384)
            return result


def mark_only_lora_as_trainable(model: nn.Module, train_bias: str = 'none', train_head: bool = False) -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        train_bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.
        train_head: if True, head (lm_head, or scalar_head) weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ['none', 'lora_only', 'all']
    """

    if train_bias not in ['none', 'lora_only', 'all']:
        raise NotImplementedError

    # freeze all layers except LoRA's, or output head layer
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            if train_head and ('lm_head' in n or 'scalar_head' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if train_bias == 'none':
        return
    elif train_bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif train_bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True


def lora_state_dict(model: nn.Module, train_bias: str = 'none', train_head: bool = False) -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        model: model with LoRA layers
        train_bias:
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.
        train_head: if True, state dict will contain weights for head (lm_head, or scalar_head).

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ['none', 'lora_only', 'all']
    """

    if train_bias not in ['none', 'lora_only', 'all']:
        raise NotImplementedError

    state_dict = model.state_dict()
    return lora_state_dict_from_full_state_dict(state_dict, train_bias, train_head)


def lora_state_dict_from_full_state_dict(
    state_dict: dict, train_bias: str = 'none', train_head: bool = False
) -> Dict[str, torch.Tensor]:
    """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

    Args:
        state_dict: nn.Module full state dict with LoRA weights
        train_bias:
            ``"none"``: state dict will not store bias weights,
            ``"lora_only"``: state dict will store bias weights only from LoRA layers,
            ``"all"``: state dict will store all bias weights.
        train_head: if True, state dict will contain weights for head (lm_head, or scalar_head).

    Returns:
        Weights and biases of LoRA layers

    Raises:
        NotImplementedError: if `bias` not in ['none', 'lora_only', 'all']
    """

    if train_bias not in ['none', 'lora_only', 'all']:
        raise NotImplementedError

    if train_bias == 'none':
        return {
            k: state_dict[k]
            for k in state_dict
            if 'lora_' in k
            or ('reward_mean' in k or 'reward_var' in k)
            or (train_head and ('lm_head' in k or 'scalar_head' in k))
        }
    elif train_bias == 'all':
        return {
            k: state_dict[k]
            for k in state_dict
            if 'lora_' in k
            or 'bias' in k
            or ('reward_mean' in k or 'reward_var' in k)
            or (train_head and ('lm_head' in k or 'scalar_head' in k))
        }
    elif train_bias == 'lora_only':
        to_return = {}
        for k in state_dict:
            if 'lora_' in k:
                to_return[k] = state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]
            elif 'reward_mean' in k or 'reward_var' in k:
                to_return[k] = state_dict[k]
            elif train_head and ('lm_head' in k or 'scalar_head' in k):
                to_return[k] = state_dict[k]

        return to_return


@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0


class Attention(llama.Attention):
    lora_args = None

    def __init__(self, args: llama.ModelArgs) -> None:
        """Attention with training q, v weights using Low Ranking Adaptation for
        parameter-efficient fine-tuning, and keep k, o fixed.

        Args:
            args:
                ``"max_seq_len"``: size of the context of the model,
                ``"vocab_size"``: number of unique tokens,
                ``"n_layers"``: number of transformer blocks (self-attention + MLP),
                ``"n_heads"``: number of heads in multi-head attention mechanism,
                ``"dim"``: size of the embedding: vector representation of each token.
        """
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            r=self.lora_args.r,
            lora_alpha=self.lora_args.alpha,
            lora_dropout=self.lora_args.dropout,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = Linear(
            args.dim,
            self.n_heads * self.head_dim,
            r=self.lora_args.r,
            lora_alpha=self.lora_args.alpha,
            lora_dropout=self.lora_args.dropout,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.use_cache = args.use_cache

        self.cache_k = None
        self.cache_v = None

        # regularization
        self.attn_dropout = nn.Dropout(args.attn_dropout) if args.attn_dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(args.resid_dropout) if args.resid_dropout > 0 else nn.Identity()


@contextmanager
def lora(r, alpha, dropout, enabled: bool = True):
    """Apply context manager under which you can instantiate the model with LoRA.

    In a nutshell the code inside this function forces to use LoRA variant of attention
    instead of the original one (without LoRA).

    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enabled: enables/disables LoRA
    """
    if not enabled:
        yield
        return

    Attention.lora_args = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    # when entering context manager replace link to Attention class from original
    # to a variant with LoRA
    attention = llama.Attention
    llama.Attention = Attention
    yield
    # when exiting context manager - restore link to original Attention class
    llama.Attention = attention

    Attention.lora_args = None