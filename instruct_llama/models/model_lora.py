# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
LLaMA model with LoRALiner layers
"""
import logging
from functools import partial
from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, Iterable, Dict

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import instruct_llama.models.model as llama
from instruct_llama.models.lora import LoRALinear, LoRAMoELinear#, LoRALinear4bit, Linear4bit
from torch.distributions import Poisson

logger = logging.getLogger(__name__)


@dataclass
class LoraModelArgs(llama.ModelArgs):
    MoE: bool = True
    n_MoE_exp: int = 128
    n_MoE_k: int = 8
    n_gnn_layers: int = 10 #layers number of gcn
    thresholds: float = 0.75 # edge connection thresholds
    lr_route: float = 0.01
    dim_gcn: int = 1024

    lora_r: float = 0.0
    lora_scaling: float = 1.0
    lora_dropout: float = 0.0

    lora_attn_query: bool = False  # train Attention query layer
    lora_attn_key: bool = False  # train Attention key layer
    lora_attn_value: bool = False  # train Attention value layer
    lora_attn_proj: bool = False  # train Attention output projection layer
    lora_attn_mlp: bool = False  # train Attention MLP block
    lora_head: bool = False  # train model output layer

    quant_4bit: bool = False  # quantize frozen linear layer
    quant_lora_4bit: bool = False  # quantize LoRA linear layer
    quant_4bit_double: bool = False
    quant_4bit_type: str = 'nf4'
    quant_compute_dtype: torch.dtype = torch.bfloat16


def _get_lora_kwargs(params: LoraModelArgs) -> Dict:
    return {
        'r': params.lora_r,
        'lora_scaling': params.lora_scaling,
        'lora_dropout': params.lora_dropout,
    }


def _get_quant_kwargs(params: LoraModelArgs) -> Dict:
    return {
        'compress_statistics': params.quant_4bit_double,
        'quant_type': params.quant_4bit_type,
        'compute_dtype': params.quant_compute_dtype,
    }

def _get_moe_kwsrgs(params: LoraModelArgs) -> Dict:
    return {
        'n': params.n_MoE_exp,
        'k': params.n_MoE_k,
        # 'perc': params.r_MoE_k,
        'r': params.lora_r,
        'lora_scaling': params.lora_scaling,
        'lora_dropout': params.lora_dropout,
    }

def _get_lora_linear_layer(params: LoraModelArgs) :#-> Union[LoRALinear, LoRALinear4bit]:
    layer_cls = None
    if params.MoE:
        kwargs = _get_moe_kwsrgs(params)
        layer_cls = LoRAMoELinear
    else:
        kwargs = _get_lora_kwargs(params)
        layer_cls = LoRALinear

    return partial(layer_cls, **kwargs)


def _get_linear_layer(params: LoraModelArgs):# -> Union[nn.Linear, Linear4bit]:
    layer_cls = None
    kwargs = {}
    layer_cls = nn.Linear

    return partial(layer_cls, **kwargs)


def get_linear_layer(params: LoraModelArgs, use_lora: bool = False):# -> Union[nn.Linear, Linear4bit, LoRALinear, LoRALinear4bit]:
    if use_lora:
        return _get_lora_linear_layer(params)
    else:
        return _get_linear_layer(params)


class Attention(llama.Attention):
    def __init__(self, args: LoraModelArgs):
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        query_layer_cls = get_linear_layer(args, args.lora_attn_query)
        key_layer_cls = get_linear_layer(args, args.lora_attn_key)
        value_layer_cls = get_linear_layer(args, args.lora_attn_value)
        proj_layer_cls = get_linear_layer(args, args.lora_attn_proj)

        self.wq = query_layer_cls(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )

        self.wk = key_layer_cls(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = value_layer_cls(
            args.dim,
            self.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = proj_layer_cls(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.use_cache = args.use_cache

        self.cache_k = None
        self.cache_v = None
        if self.use_cache:
            self.cache_k = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_heads, self.head_dim))
            self.cache_v = torch.zeros((self.max_batch_size, self.max_seq_len, self.n_heads, self.head_dim))

        # regularization
        self.attn_dropout = nn.Dropout(args.attn_dropout) if args.attn_dropout > 0 else nn.Identity()


class moegnn(nn.Module):
    def __init__(self, dim, n_exp, n_layers, thresholds, k, dim_gcn):
        super().__init__()
        self.n_exp = n_exp
        self.convs = nn.ModuleList()
        self.X = torch.rand(dim, n_exp, device = 'cuda') #初始特征
        self.thresholds = thresholds
        self.k = k
        self.convs.append(GCNConv(dim, dim_gcn, bias = False))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(dim_gcn, dim_gcn, bias = False))
        self.convs.append(GCNConv(dim_gcn, dim, bias = False))
        self.mlp = nn.Linear(dim, dim, bias = False)
        self.nl = nn.ReLU()
        self.lamb = nn.Parameter(torch.tensor(1.5, requires_grad = True))
        self.miu = n_exp / 2
        self.theta = nn.Parameter(torch.tensor([1.0]))
        self.proj = nn.Linear(dim, 1, bias = False)
        self.dim = dim

    def get_edges(self, X):
        edges = torch.empty(2, 0, device = X.device) # 生成图

        for i in range(self.n_exp):
            col = torch.tensor([[self.n_exp], [i]], device = X.device)
            edges = torch.cat((edges, col),dim = 1)
            for j in range(i+1, self.n_exp):
                if(F.cosine_similarity(X[:, i].unsqueeze(0), X[:, j].unsqueeze(0)) > self.thresholds):
                    col = torch.tensor([[i], [j]], device = X.device) # 一条边，节点pair形式
                    edges = torch.cat((edges, col), dim=1)
        return edges

    def forward(self, x):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        ori_shape = x.shape[:-1]
        x = self.nl(self.mlp(x)) # linear project
        exp = self.nl(self.mlp(self.X.T))
        exp = exp.T

        edge_index = self.get_edges(exp)
        edge_index = edge_index.long()
        edge_index = edge_index.to('cuda')

        x = x.view(-1, self.dim)
        results = []
        n_loop = x.shape[0]
        for i in range(n_loop):
            t = x[i]
            t = t.unsqueeze(1)
            t = torch.cat((exp, t), dim = 1)

            t = self.convs[0](t.T, edge_index)
            for conv in self.convs[1:-1]:
                t = conv(t, edge_index)
                t = torch.relu(t)
            t = self.convs[-1](t, edge_index)

            t = self.proj(t).squeeze()
            t = t[:-1]
            t = F.softmax(t, dim = 0)
            results.append(t)
        
        result = torch.stack(results, dim = 0)

        r_shape = (*ori_shape, self.n_exp)
        result = result.reshape(r_shape)
        return result


class FeedForward(llama.FeedForward):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        resid_dropout: Optional[float],
        args: LoraModelArgs,
        MoE: bool,
    ):
        nn.Module.__init__(self)
        self.args = args

        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        layer_cls = get_linear_layer(args, args.lora_attn_mlp)
        self.w1 = layer_cls(dim, hidden_dim, bias=False)
        self.w2 = layer_cls(hidden_dim, dim, bias=False)
        self.w3 = layer_cls(dim, hidden_dim, bias=False)
        self.gnn = moegnn(args.dim, args.n_MoE_exp, args.n_gnn_layers, args.thresholds, args.n_MoE_k, args.dim_gcn)
        self.miu = self.gnn.miu
        self.theta = self.gnn.theta
        
        # regularization
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout > 0 else nn.Identity()
        
        
    def forward(self, x):
        eps = 1e-5
        alloc = self.gnn(x) #get the weight for experts
        lamb = self.gnn.lamb
        if lamb < 1:
            lamb = 1
        if lamb > 2:
            lamb = 2
        s_alloc, _ = torch.sort(alloc, dim = -1, descending=True)
        poisson_lamb = Poisson(rate = lamb)
        pois_probs = poisson_lamb.log_prob(torch.arange(1, self.args.n_MoE_exp + 1, device = x.device).float() )
        self.route_loss = torch.sum(torch.exp(pois_probs) * (pois_probs - torch.log(s_alloc + eps) ), dim = -1) # ...,n_exp
        self.route_loss = torch.sum(self.route_loss) / (self.route_loss.numel() + eps)
        values, indices = torch.topk(alloc, self.gnn.k, dim = -1)
        
        self.count = torch.zeros_like(alloc)
        src = torch.ones_like(indices, dtype = self.count.dtype)
        self.count.scatter_(dim=2, index=indices, src=src)
        self.count = torch.sum(self.count, dim = (-1) ) # top-K count(n_exp)

        alloc = torch.zeros_like(alloc)
        alloc = alloc.scatter_(dim = -1, index=indices, src=values) # only top-k (..., n_exp)

        output = self.w2(F.silu(self.w1(x, alloc)) * self.w3(x, alloc), alloc)
        output = self.resid_dropout(output)
        return output



class TransformerBlock(llama.TransformerBlock):
    def __init__(self, layer_id: int, args: LoraModelArgs):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            resid_dropout=args.resid_dropout,
            args=args,
            MoE = args.MoE
        )
        self.attention_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = llama.RMSNorm(args.dim, eps=args.norm_eps)


class Transformer(llama.Transformer):
    def __init__(self, params: LoraModelArgs):
        nn.Module.__init__(self)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.token_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.embed_dropout = nn.Dropout(params.embed_dropout) if params.embed_dropout > 0 else nn.Identity()
        self.head_dropout = nn.Dropout(params.head_dropout) if params.head_dropout > 0 else nn.Identity()

        self.layers: Iterable[TransformerBlock] = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.post_norm = llama.RMSNorm(params.dim, eps=params.norm_eps)

        head_layer_cls = get_linear_layer(params, params.lora_head)
        if self.params.head_type == 'lm_head':
            logger.info('Creating LLaMA-2 model with LM head ...')
            self.lm_head = head_layer_cls(params.dim, params.vocab_size, bias=False)
        elif self.params.head_type == 'scalar_head':
            logger.info('Creating LLaMA-2 model with scalar head ...')
            self.scalar_head = head_layer_cls(params.dim, 1, bias=False)
        elif self.params.head_type == 'dual_head':
            logger.info('Creating LLaMA-2 model with LM and scalar heads ...')
            self.lm_head = head_layer_cls(params.dim, params.vocab_size, bias=False)
            self.scalar_head = head_layer_cls(params.dim, 1, bias=False)

        self.freqs_cis = llama.precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)
