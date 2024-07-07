# @Time    : 31/01/2024 15:16
# @Author  : BubblyYi
# @FileName: unixvit.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.runner import auto_fp16
from mmdet.models.builder import BACKBONES
from timm.models.layers import trunc_normal_

from .dualvit import (PVT2FFN, Attention, DownSamples, DualAttention,
                      DualBlock, DualVit, DWConv, MergeBlock, MergeFFN,
                      SemanticEmbed, Stem)


def get_masked_attn_output_weights(attn_weights,
                                   bsz,
                                   tgt_len,
                                   src_len,
                                   attn_mask=None,
                                   key_padding_mask=None,
                                   num_heads=1):

    attn_weights_org_size = attn_weights.size()
    if list(attn_weights_org_size) != [bsz * num_heads, tgt_len, src_len]:
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    assert list(attn_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, tgt_len, src_len]:
                print('attn_mask size:', attn_mask.size(), flush=True)
                print('[1, tgt_len, src_len]:', [1, tgt_len, src_len],
                      flush=True)
                raise RuntimeError(
                    'The size of the 2D attn_mask is not correct.')
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dtype == torch.bool:
            attn_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_weights += attn_mask

    if key_padding_mask is not None:

        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)
    if list(attn_weights_org_size) != [bsz * num_heads, tgt_len, src_len]:
        attn_weights = attn_weights.view(attn_weights_org_size)

    return attn_weights


class ScoreNet(nn.Module):

    def __init__(self, dim, num_scores):
        super(ScoreNet, self).__init__()
        self.dim = dim
        self.score_net = nn.Sequential(nn.Linear(self.dim, self.dim // 2),
                                       nn.LayerNorm(self.dim // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.dim // 2, num_scores),
                                       nn.Sigmoid())

    def forward(self, x):
        x = self.score_net(x).permute(2, 0, 1)
        return x


class DualMaskedAttention(DualAttention):

    def selfatt(self, semantics, key_padding_mask=None, attn_mask=None):

        B, N, C = semantics.shape
        qkv = self.qkv_proxy(semantics).reshape(B, -1, 3, self.num_heads,
                                                C // self.num_heads).permute(
                                                    2, 0, 3, 1, 4)
        attn = (qkv[0] @ qkv[1].transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        semantics = (attn @ qkv[2]).transpose(1, 2).reshape(B, N, C)
        return semantics

    def forward(self,
                x,
                H,
                W,
                semantics,
                key_padding_mask=None,
                attn_mask=None):

        semantics = semantics + self.drop_path(
            self.gamma1 * self.selfatt(semantics))

        B, N, C = x.shape
        B_p, N_p, C_p = semantics.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        q_semantics = self.q_proxy(self.q_proxy_ln(semantics)).reshape(
            B_p, N_p, self.num_heads,
            C // self.num_heads).permute(0, 2, 1, 3)
        kv_semantics = self.kv_proxy(x).reshape(B, -1, 2, self.num_heads,
                                                C // self.num_heads).permute(
                                                    2, 0, 3, 1, 4)

        bsz, num_heads, tgt_len, _ = q_semantics.shape
        _, _, src_len, _ = kv_semantics[0].shape

        attn = (q_semantics @ kv_semantics[0].transpose(-2, -1)) * self.scale
        attn = get_masked_attn_output_weights(
            attn_weights=attn,
            bsz=bsz,
            num_heads=num_heads,
            tgt_len=tgt_len,
            src_len=src_len,
            key_padding_mask=key_padding_mask)

        attn = attn.softmax(dim=-1)

        semantics = semantics + self.drop_path(
            (attn @ kv_semantics[1]).transpose(1, 2).reshape(B, N_p, C) *
            self.gamma2)

        semantics = semantics + self.drop_path(
            self.gamma3 * self.mlp_proxy(self.p_ln(semantics)))

        kv = self.kv(self.proxy_ln(semantics)).reshape(
            B, -1, 2, self.num_heads,
            C // self.num_heads).permute(2, 0, 3, 1, 4)
        bsz, num_heads, tgt_len, _ = q.shape
        _, _, src_len, _ = kv[0].shape

        attn = (q @ kv[0].transpose(-2, -1)) * self.scale
        attn = get_masked_attn_output_weights(attn_weights=attn,
                                              bsz=bsz,
                                              num_heads=num_heads,
                                              tgt_len=tgt_len,
                                              src_len=src_len,
                                              attn_mask=None,
                                              key_padding_mask=None)
        attn = attn.softmax(dim=-1)
        x = (attn @ kv[1]).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, semantics


class DualBlockMaskedEmbed(DualBlock):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 with_cp=False):
        super(DualBlockMaskedEmbed, self).__init__(dim=dim,
                                                   num_heads=num_heads,
                                                   mlp_ratio=mlp_ratio,
                                                   drop_path=drop_path,
                                                   norm_layer=norm_layer,
                                                   with_cp=with_cp)
        self.mlp = PVT2FFNEmbed(in_features=dim,
                                hidden_features=int(dim * mlp_ratio))
        self.attn = DualMaskedAttention(dim, num_heads, drop_path=drop_path)

    def forward(self,
                x,
                H,
                W,
                semantics,
                num_extra_token,
                key_padding_mask=None,
                attn_mask=None):

        def inner_forward(x, semantics, H, W):
            _x, semantics = self.attn(self.norm1(x),
                                      H,
                                      W,
                                      semantics,
                                      key_padding_mask=key_padding_mask,
                                      attn_mask=attn_mask)
            x = x + self.drop_path(self.gamma1 * _x)
            x = x + self.drop_path(
                self.gamma2 * self.mlp(self.norm2(x), H, W, num_extra_token))
            return x, semantics

        if self.with_cp:
            x, semantics = checkpoint.checkpoint(
                inner_forward,
                x,
                semantics,
                H,
                W,
            )
        else:
            x, semantics = inner_forward(x, semantics, H, W)
        return x, semantics


class MaskedAttention(Attention):

    def forward(self, x, H, W, key_padding_mask=None, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        bsz, num_heads, tgt_len, _ = q.shape
        src_len = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = get_masked_attn_output_weights(
            attn,
            bsz,
            tgt_len,
            src_len,
            num_heads=num_heads,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PVT2FFNEmbed(PVT2FFN):

    def __init__(self, in_features, hidden_features):
        super(PVT2FFNEmbed, self).__init__(in_features=in_features,
                                           hidden_features=hidden_features)
        self.dwconv = DWConv(hidden_features)

    def forward(self, x, H, W, num_extra_token):
        x = self.fc1(x)

        extra_token = x[:, :num_extra_token, :]
        x = x[:, num_extra_token:num_extra_token + H * W * 2, :]

        x = x.chunk(2, dim=1)
        x = torch.cat([x[0], x[1]], dim=0)
        x = self.dwconv(x, H, W)
        x = x.chunk(2, dim=0)
        x = torch.cat([x[0], x[1]], dim=1)

        x = torch.cat([extra_token, x], dim=1)
        # 处理B 2N C --> 2B N C--> B 2N C
        x = self.act(x)
        x = self.fc2(x)
        return x


class MergeFFNEmbed(MergeFFN):

    def forward(self, x, H, W, num_extra_token):
        semantics = x[:, num_extra_token + H * W * 2:, :]
        semantics = self.fc_proxy(semantics)

        x = self.fc1(x)
        extra_token = x[:, :num_extra_token, :]
        x = x[:, num_extra_token:num_extra_token + H * W * 2, :]

        x = x.chunk(2, dim=1)
        x = torch.cat([x[0], x[1]], dim=0)
        x = self.dwconv(x, H, W)
        x = x.chunk(2, dim=0)
        x = torch.cat([x[0], x[1]], dim=1)

        x = torch.cat([extra_token, x], dim=1)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.cat([x, semantics], dim=1)
        return x


class MergeBlockMaskedEmbed(MergeBlock):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 with_cp=False):
        super(MergeBlockMaskedEmbed, self).__init__(dim=dim,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio,
                                                    drop_path=drop_path,
                                                    norm_layer=norm_layer,
                                                    is_last=is_last,
                                                    with_cp=with_cp)
        self.attn = MaskedAttention(dim, num_heads)
        if is_last:
            self.mlp = PVT2FFNEmbed(in_features=dim,
                                    hidden_features=int(dim * mlp_ratio))
        else:
            self.mlp = MergeFFNEmbed(in_features=dim,
                                     hidden_features=int(dim * mlp_ratio))

    def forward(self,
                x,
                H,
                W,
                num_extra_token,
                key_padding_mask=None,
                attn_mask=None):

        def innerforword(x):

            x = x + self.drop_path(
                self.gamma1 * self.attn(self.norm1(x),
                                        H,
                                        W,
                                        key_padding_mask=key_padding_mask,
                                        attn_mask=attn_mask))
            if self.is_last:
                x = x[:, :2 * H * W + num_extra_token]
                x = x + self.drop_path(self.gamma2 * self.mlp(
                    self.norm2(x), H, W, num_extra_token))
            else:
                x = x + self.drop_path(self.gamma2 * self.mlp(
                    self.norm2(x), H, W, num_extra_token))
            return x

        if self.with_cp:
            x = checkpoint.checkpoint(innerforword, x)
        else:
            x = innerforword(x)
        return x


@BACKBONES.register_module()
class UNIXVit(DualVit):

    def __init__(self,
                 stem_width=32,
                 in_chans=3,
                 embed_dims=[64, 128, 320, 448],
                 num_heads=[2, 4, 10, 14],
                 mlp_ratios=[8, 8, 4, 3],
                 drop_path_rate=0.15,
                 norm_layer='LN',
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 score_embed_nums=1,
                 num_scores=2,
                 mod_nums=1,
                 with_cp=False,
                 pretrained=None):
        super().__init__()

        self.pool_ir = nn.AvgPool2d((7, 7), stride=7)

        if norm_layer == 'LN':
            norm_layer = nn.LayerNorm
        self.with_cp = with_cp
        self.depths = depths
        self.num_stages = num_stages

        self.sep_stage = 2
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_width, embed_dims[i])
                patch_embed_ir = Stem(in_chans, stem_width, embed_dims[i])

            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])
                patch_embed_ir = DownSamples(embed_dims[i - 1],
                                             embed_dims[i])

            if i == 0:
                self.q = nn.Parameter(torch.empty((64, embed_dims[0])),
                                      requires_grad=True)
                self.q_embed = nn.Sequential(
                    nn.LayerNorm(embed_dims[0]),
                    nn.Linear(embed_dims[0], embed_dims[0]))
                self.pool = nn.AvgPool2d((7, 7), stride=7)
                self.kv = nn.Linear(embed_dims[0], 2 * embed_dims[0])
                self.scale = embed_dims[0]**-0.5
                self.proxy_ln = nn.LayerNorm(embed_dims[0])
                self.se = nn.Sequential(
                    nn.Linear(embed_dims[0], embed_dims[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims[0], 2 * embed_dims[0]))
                trunc_normal_(self.q, std=.02)
            else:
                semantic_embed = SemanticEmbed(embed_dims[i - 1],
                                               embed_dims[i])
                setattr(self, f'proxy_embed{i + 1}', semantic_embed)

            if i >= self.sep_stage:
                block = nn.ModuleList([
                    MergeBlockMaskedEmbed(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i] - 1 if
                        (j % 2 != 0 and i == 2) else mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        is_last=((i == 3) and (j == depths[i] - 1)),
                        with_cp=with_cp) for j in range(depths[i])
                ])
            else:
                block = nn.ModuleList([
                    DualBlockMaskedEmbed(dim=embed_dims[i],
                                         num_heads=num_heads[i],
                                         mlp_ratio=mlp_ratios[i],
                                         drop_path=dpr[cur + j],
                                         norm_layer=norm_layer,
                                         with_cp=with_cp)
                    for j in range(depths[i])
                ])

            norm = norm_layer(embed_dims[i])
            norm_proxy = norm_layer(embed_dims[i])
            out_norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'patch_embed_ir{i + 1}', patch_embed_ir)

            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
            setattr(self, f'out_norm{i + 1}', out_norm)

            if i != num_stages - 1:
                setattr(self, f'norm_proxy{i + 1}', norm_proxy)

        self.mod_nums = mod_nums
        self.mod_embed = None
        self.score_embed = None

        self.mod_embeds = nn.ParameterList([
            nn.Parameter(torch.rand([1, self.mod_nums, embed_dims[i]]))
            for i in range(num_stages)
        ])

        self.score_embed_nums = score_embed_nums
        self.num_scores = num_scores

        self.score_embeds = nn.ParameterList([
            nn.Parameter(
                torch.rand([1, self.score_embed_nums, embed_dims[i]]))
            for i in range(num_stages)
        ])

        self.scorenet = nn.ModuleList([
            ScoreNet(dim=embed_dims[i], num_scores=self.num_scores)
            for i in range(num_stages)
        ])
        self.extra_token_num = self.mod_nums + self.score_embed_nums

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def get_fist_semantics(self, x, x_ir, B, H, W, C, x_weight, x_ir_weight):
        x_down = self.pool(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        x_down_H, x_down_W = x_down.shape[2:]
        x_down = x_down.view(B, C, -1).permute(0, 2, 1)
        x_ir_down = self.pool_ir(
            x_ir.reshape(B, H, W, C).permute(0, 3, 1, 2))
        x_ir_down = x_ir_down.view(B, C, -1).permute(0, 2, 1)

        x_down = torch.cat([x_down, x_ir_down], dim=1)
        kv = self.kv(x_down).view(B, -1, 2, C).permute(2, 0, 1, 3)

        attn_self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1)
        attn_self_q = F.interpolate(
            attn_self_q.unsqueeze(0),
            size=(x_down_H, x_down_W),
            mode='bicubic').squeeze(0).permute(1, 2, 0)

        attn_self_q = attn_self_q.reshape(-1, attn_self_q.shape[-1])
        attn_self_q = self.q_embed(attn_self_q)
        bsz, src_len, _ = kv[0].shape

        N = x_down.shape[1]
        M = attn_self_q.shape[0]
        key_padding_mask = torch.zeros((B, N), dtype=torch.bool).to(x.device)
        attn_mask = torch.zeros((M, N), dtype=torch.bool).to(x.device)

        total_patches = N // 2
        if x_weight == 1 and x_ir_weight == 0:
            key_padding_mask[:, total_patches:] = True
            attn_mask[:, total_patches:] = True

        elif x_weight == 0 and x_ir_weight == 1:
            key_padding_mask[:, :total_patches] = True
            attn_mask[:, :total_patches] = True

        attn_self_q = (attn_self_q @ kv[0].transpose(-1, -2)) * self.scale

        attn_self_q = get_masked_attn_output_weights(
            attn_self_q, bsz=bsz, num_heads=1, tgt_len=M, src_len=src_len,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        attn_self_q = attn_self_q.softmax(-1)  # B, M, N
        semantics = attn_self_q @ kv[1]  # B, M, C
        semantics = semantics.view(B, -1, C)

        x_down = (x_down[:, : total_patches, :] * x_weight +
                  x_down[:, total_patches:, :] * x_ir_weight) / (x_weight +
                                                                 x_ir_weight)
        semantics = torch.cat(
            [semantics.unsqueeze(2),
             x_down.unsqueeze(2)], dim=2)
        se = self.se(semantics.sum(2).mean(1)).view(B, 2,
                                                    C).softmax(1)
        semantics = (semantics * se.unsqueeze(1)).sum(2)
        semantics = self.proxy_ln(semantics)
        return semantics

    def forward_sep(self, x, x_ir, x_weight, x_ir_weight):

        B = x.shape[0]
        outs = []
        score_start_idx = 0
        score_end_idx = score_start_idx + self.score_embed_nums

        mod_emb_end_idx = score_end_idx + self.mod_nums

        for i in range(self.sep_stage):
            score_embed = self.score_embeds[i]
            mod_embed = self.mod_embeds[i]

            patch_embed = getattr(self, f'patch_embed{i + 1}')
            patch_embed_ir = getattr(self, f'patch_embed_ir{i + 1}')

            block = getattr(self, f'block{i + 1}')

            x, H, W = patch_embed(x)
            x_ir, _, _ = patch_embed_ir(x_ir)
            C = x.shape[-1]

            H_tensor = torch.tensor([H], requires_grad=False)
            W_tensor = torch.tensor([W], requires_grad=False)
            B_tensor = torch.tensor([B], requires_grad=False)
            C_tensor = torch.tensor([C], requires_grad=False)

            if i == 0:
                if self.with_cp:
                    semantics = checkpoint.checkpoint(
                        self.get_fist_semantics, x, x_ir, B_tensor, H_tensor,
                        W_tensor, C_tensor, x_weight, x_ir_weight)
                else:
                    semantics = self.get_fist_semantics(
                        x, x_ir, B, H, W, C, x_weight, x_ir_weight)
            else:

                def inner_get_semantics(semantics):
                    semantics_embed = getattr(self, f'proxy_embed{i + 1}')
                    semantics = semantics_embed(semantics)
                    return semantics

                if self.with_cp:
                    semantics = checkpoint.checkpoint(
                        inner_get_semantics, semantics)
                else:
                    semantics = inner_get_semantics(semantics)

            x = torch.cat([score_embed, mod_embed, x, x_ir], dim=1)

            N = H * W * 2 + self.extra_token_num
            key_padding_mask = torch.zeros((B, N),
                                           dtype=torch.bool).to(x.device)
            attn_mask = None
            total_patches = H * W + self.extra_token_num

            if x_weight == 1 and x_ir_weight == 0:
                key_padding_mask[:, total_patches:] = True

            elif x_weight == 0 and x_ir_weight == 1:
                key_padding_mask[:,
                                 self.extra_token_num:total_patches] = True

            for blk in block:

                if self.with_cp:
                    x, semantics = blk(x, H_tensor, W_tensor, semantics,
                                       self.extra_token_num,
                                       key_padding_mask, attn_mask)
                else:
                    x, semantics = blk(x, H, W, semantics,
                                       self.extra_token_num,
                                       key_padding_mask, attn_mask)

            norm = getattr(self, f'norm{i + 1}')
            x = norm(x)
            extra_token = x[:, :self.extra_token_num, :]
            x = x[:, self.extra_token_num:, :]
            # 有可能需要变更score_embed和mod_embed
            score_embed = extra_token[:, :score_end_idx, :]
            mod_embed = extra_token[:, score_end_idx:mod_emb_end_idx, :]

            if self.score_embed_nums > 1:
                score_embed = (
                    score_embed[:, 0, :] * x_weight + score_embed[:, 1, :] *
                    x_ir_weight / (x_weight + x_ir_weight)).unsqueeze(0)

            if self.mod_nums > 1:
                mod_embed = (mod_embed[:, 0, :] * x_weight +
                             mod_embed[:, 1, :] * x_ir_weight /
                             (x_weight + x_ir_weight)).unsqueeze(0)

            score_weight = self.scorenet[i](score_embed)
            x, x_ir = x.chunk(2, dim=1)
            x = x * score_weight[0]
            x_ir = x_ir * score_weight[1]

            out_x = ((x * x_weight + x_ir * x_ir_weight) /
                     (x_weight + x_ir_weight)) + mod_embed

            out_norm = getattr(self, f'out_norm{i + 1}')

            out_x = out_norm(out_x)

            out_x = out_x.reshape(B, H, W, -1).permute(0, 3, 1,
                                                       2).contiguous()

            outs.append(out_x)

            norm_semantics = getattr(self, f'norm_proxy{i + 1}')
            semantics = norm_semantics(semantics)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_ir = x_ir.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = torch.cat([x, x_ir], dim=0)

        return x, semantics, tuple(outs)

    def forward_merge(self, x, semantics, x_weight, x_ir_weight):
        score_start_idx = 0
        score_end_idx = score_start_idx + self.score_embed_nums
        mod_emb_end_idx = score_end_idx + self.mod_nums

        outs = []
        for i in range(self.sep_stage, self.num_stages):
            score_embed = self.score_embeds[i]
            mod_embed = self.mod_embeds[i]
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            patch_embed_ir = getattr(self, f'patch_embed_ir{i + 1}')
            block = getattr(self, f'block{i + 1}')

            x, x_ir = x.chunk(2, dim=0)
            B = x.shape[0]

            x, H, W = patch_embed(x)
            x_ir, _, _ = patch_embed_ir(x_ir)

            semantics_embed = getattr(self, f'proxy_embed{i + 1}')
            semantics = semantics_embed(semantics)

            x = torch.cat([score_embed, mod_embed, x, x_ir, semantics],
                          dim=1)

            M = semantics.shape[1]

            N = H * W * 2 + M + self.extra_token_num
            key_padding_mask = torch.zeros((B, N),
                                           dtype=torch.bool).to(x.device)
            attn_mask = torch.zeros((N, N), dtype=torch.bool).to(x.device)

            patches_start_1 = self.extra_token_num
            patches_end_1 = H * W + self.extra_token_num

            patches_start_2 = H * W + self.extra_token_num
            patches_end_2 = H * W * 2 + self.extra_token_num

            if x_weight == 1 and x_ir_weight == 0:
                key_padding_mask[:, patches_start_2:patches_end_2] = True
                attn_mask[patches_start_2:patches_end_2,
                          patches_start_2:patches_end_2] = True

            elif x_weight == 0 and x_ir_weight == 1:
                key_padding_mask[:, patches_start_1:patches_end_1] = True
                attn_mask[patches_start_1:patches_end_1,
                          patches_start_1:patches_end_1] = True

            for blk in block:
                x = blk(x, H, W, self.extra_token_num, key_padding_mask,
                        attn_mask)

            if i != self.num_stages - 1:
                semantics = x[:, patches_end_2:, :]
                norm_semantics = getattr(self, f'norm_proxy{i + 1}')
                semantics = norm_semantics(semantics)

            norm = getattr(self, f'norm{i + 1}')
            x = norm(x)
            extra_token = x[:, :self.extra_token_num, :]
            x = x[:, patches_start_1:patches_end_2, :]
            score_embed = extra_token[:, :score_end_idx, :]
            mod_embed = extra_token[:,
                                    score_end_idx: mod_emb_end_idx,
                                    :]
            if self.score_embed_nums > 1:
                score_embed = (
                    score_embed[:, 0, :] * x_weight + score_embed[:, 1, :] *
                    x_ir_weight / (x_weight + x_ir_weight)).unsqueeze(0)

            if self.mod_nums > 1:
                mod_embed = (mod_embed[:, 0, :] * x_weight +
                             mod_embed[:, 1, :] * x_ir_weight /
                             (x_weight + x_ir_weight)).unsqueeze(0)

            score_weight = self.scorenet[i](score_embed)
            x_rgb, x_ir = x.chunk(2, dim=1)
            x_rgb = x_rgb * score_weight[0]
            x_ir = x_ir * score_weight[1]

            out_x = ((x_rgb * x_weight + x_ir * x_ir_weight) /
                     (x_weight + x_ir_weight)) + mod_embed

            out_norm = getattr(self, f'out_norm{i + 1}')

            out_x = out_norm(out_x)

            out_x = out_x.reshape(B, H, W, -1).permute(0, 3, 1,
                                                       2).contiguous()
            outs.append(out_x)

            x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1,
                                                       2).contiguous()
            x_ir = x_ir.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = torch.cat([x_rgb, x_ir], dim=0)

        outs = tuple(outs)
        return outs

    @auto_fp16()
    def forward(self, x, x_ir):

        x_weight = torch.tensor([1], requires_grad=False, device=x.device)
        x_ir_weight = torch.tensor([1], requires_grad=False, device=x.device)

        with torch.no_grad():

            flage_x = torch.sum(x.detach())
            flage_x_ir = torch.sum(x_ir.detach())

            if flage_x == 0:
                x_weight = 0 * x_weight
            if flage_x_ir == 0:
                x_ir_weight = 0 * x_ir_weight

        x, semantics, out1 = self.forward_sep(x, x_ir, x_weight, x_ir_weight)

        def inner_forward(x, semantics, x_weight, x_ir_weight):
            out2 = self.forward_merge(
                x, semantics, x_weight=x_weight, x_ir_weight=x_ir_weight)
            return out2

        if self.with_cp:
            out2 = checkpoint.checkpoint(
                inner_forward, x, semantics, x_weight, x_ir_weight)
        else:
            out2 = inner_forward(x, semantics, x_weight, x_ir_weight)
        outs = out1 + out2
        return outs
