import torch
from torch import nn
from functools import partial
from typing import Optional, Tuple
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch import Tensor
import torch.nn.functional as F

from .utils import pytorch_utils as pt_utils


NORM_DICT = {
    "batch_norm": nn.BatchNorm1d,
    "id": nn.Identity,
    "layer_norm": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


class SharedMultiHeadAttention(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(SharedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
        self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        self.v2_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.out_proj2 = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj_weight)
        xavier_uniform_(self.k_proj_weight)
        xavier_uniform_(self.v_proj_weight)
        xavier_uniform_(self.v2_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
            constant_(self.out_proj2.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiHeadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(SharedMultiHeadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, value2: Tensor = None, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:

        v1 = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight)

        if value2 is None:
            return v1
        else:
            v2 = F.multi_head_attention_forward(
                query, key, value2, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj2.weight, self.out_proj2.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v2_proj_weight)
            return v1, v2


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.s_attn = SharedMultiHeadAttention(
            cfg.feat_dim, cfg.num_heads, cfg.attn_dropout)
        self.attn = SharedMultiHeadAttention(
            cfg.feat_dim, cfg.num_heads, cfg.attn_dropout)

        self.mem_ffn = (
            pt_utils.Seq(cfg.feat_dim)
            .batchnorm1d()
            .relu()
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(cfg.feat_dim, activation=None)
        )

        self.pre_norm1 = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.pre_norm1_mask = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.pre_norm2 = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.pre_norm2_mask = NORM_DICT[cfg.norm](cfg.feat_dim)

        # self.pre_proto_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.pre_mem_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
        self.s_mask_emb = (
            pt_utils.Seq(1)
            .conv1d(cfg.feat_dim, activation=None)
        )

        if cfg.ffn_cfg:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.ffn_cfg.hidden_dim,
                          bias=cfg.ffn_cfg.use_bias),
                ACTIVATION_DICT[cfg.ffn_cfg.activation](),
                nn.Dropout(cfg.ffn_cfg.dropout, inplace=True),
                nn.Linear(cfg.ffn_cfg.hidden_dim, cfg.feat_dim,
                          bias=cfg.ffn_cfg.use_bias)

            )
            self.pre_ffn_norm = NORM_DICT[cfg.ffn_cfg.norm](cfg.feat_dim)
            self.ffn_dropout = nn.Dropout(cfg.ffn_cfg.dropout, inplace=True)

            self.ffn_mask = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.ffn_cfg.hidden_dim,
                          bias=cfg.ffn_cfg.use_bias),
                ACTIVATION_DICT[cfg.ffn_cfg.activation](),
                nn.Dropout(cfg.ffn_cfg.dropout, inplace=True),
                nn.Linear(cfg.ffn_cfg.hidden_dim, cfg.feat_dim,
                          bias=cfg.ffn_cfg.use_bias)

            )
            self.pre_ffn_norm_mask = NORM_DICT[cfg.ffn_cfg.norm](cfg.feat_dim)
            self.ffn_dropout_mask = nn.Dropout(
                cfg.ffn_cfg.dropout, inplace=True)

        if cfg.pos_emb_cfg:
            self.q_pos_emb = (
                pt_utils.Seq(3)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, activation=None)
            )
            self.m_pos_emb = (
                pt_utils.Seq(3)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, activation=None)
            )
            self.pos_emb = (
                pt_utils.Seq(3)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, activation=None)
            )

        # self.l_dropout = nn.Dropout(cfg.dropout)
        self.s_dropout = nn.Dropout(cfg.dropout)
        self.s_dropout_mask = nn.Dropout(cfg.dropout)

        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout_mask = nn.Dropout(cfg.dropout)

        self.cfg = cfg

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def with_mask_embed(self, tensor, mask):
        return tensor if mask is None else tensor + mask

    def forward(self, input):

        feat = input.pop('feat')  # b,c,n
        xyz = input.pop('xyz')  # b,n,3
        m_feat = input.pop('m_feat')  # b,c,t,n
        m_xyz = input.pop('m_xyz')  # b,t,n,3
        batch_size = feat.shape[0]

        x = feat.permute(2, 0, 1)
        norm_x = self.pre_norm2(x)

        # short-term memory
        if m_feat is None:
            mem_feat_prop = mem_feat = self.mem_ffn(feat)  # b,c,n
            mem_xyz = xyz
        else:
            if len(m_feat.shape) == 4:
                # b,c,t,n
                _, _, memory_size, npts = m_feat.shape
                mem_feat = m_feat.reshape(
                    batch_size, -1, memory_size*npts)
                mem_xyz = m_xyz.reshape(batch_size, memory_size*npts, -1)
            else:
                mem_feat = m_feat
                mem_xyz = m_xyz
            # print(mem_feat.shape, mem_xyz.shape, mem_mask.shape)
            mem_feat_prop = self.mem_ffn(feat)

        mem_norm = self.pre_mem_norm(mem_feat.permute(2, 0, 1))
        if self.cfg.pos_emb_cfg:
            q_pe = xyz.permute(0, 2, 1).contiguous()
            q_pe = self.q_pos_emb(q_pe).permute(2, 0, 1)
            m_pe = mem_xyz.permute(0, 2, 1).contiguous()
            m_pe = self.m_pos_emb(m_pe).permute(2, 0, 1)
        else:
            q_pe = m_pe = None

        q_s = self.with_pos_embed(norm_x, q_pe)
        k_s = self.with_pos_embed(mem_norm, m_pe)
        v_s = mem_norm

        (xx, _) = self.s_attn(q_s, k_s, v_s)
        x = x + self.s_dropout(xx)

        # self-attn
        if self.cfg.pos_emb_cfg:
            pe = xyz.permute(0, 2, 1).contiguous()
            pe = self.pos_emb(pe).permute(2, 0, 1)
        else:
            pe = None

        xx = self.pre_norm1(x)
        q = k = self.with_pos_embed(xx, pe)
        v = xx
        (xx, _) = self.attn(q, k, v, attn_mask=None)
        x = x + self.dropout(xx)

        if self.cfg.ffn_cfg:
            xx = self.pre_ffn_norm(x)
            xx = self.ffn(xx)
            x = x + self.ffn_dropout(xx)

        feat = x.permute(1, 2, 0)

        if self.cfg.is_last:
            output_dict = dict(
                geo_feat=feat,
                xyz=xyz,
                mem_feat_prop=mem_feat_prop,
                mem_feat=mem_feat,
                mem_xyz=mem_xyz
            )
        else:
            output_dict = dict(
                feat=feat,
                xyz=xyz,
                mem_feat_prop=mem_feat_prop,
                mem_feat=mem_feat,
                mem_xyz=mem_xyz
            )

        return output_dict


class FusionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.ModuleList()
        for idx, layer_cfg in enumerate(cfg.layers_cfg):
            layer_cfg.is_first = (idx == 0)
            layer_cfg.is_last = (idx == len(cfg.layers_cfg)-1)
            self.layers.append(TransformerLayer(layer_cfg))

        self.cfg = cfg

    def forward(self, input_dict):

        m_feat = input_dict.pop('m_feat')
        m_xyz = input_dict.pop('m_xyz')
        output_dict = dict()

        feats = []
        for i, layer in enumerate(self.layers):
            input_dict.update(
                m_feat=m_feat,
                m_xyz=m_xyz
            )
            input_dict = layer(input_dict)
            feats.append(input_dict['mem_feat_prop'])
        feats = torch.stack(feats, dim=0)
        # nl, b, c, n

        output_dict.update(input_dict)
        output_dict.update(
            layer_feats=feats
        )
        return output_dict
