# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention
from fastvideo.utils.communications import gather_sequence,split_sequence,all_to_all_4D
__all__ = ['WanModel']

from .audio_adapters import AudioProjNet2,PerceiverAttentionCA,AudioAttentionFA
def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


# @amp.autocast(enabled=False)
# def rope_apply(x, grid_sizes, freqs,nccl_info):
#     """
#     x:          [B, L, N, C].
#     grid_sizes: [B, 3].
#     freqs:      [M, C // 2].
#     """
#     s, n, c = x.size(1), x.size(2), x.size(3) // 2
#     # split freqs
#     freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
#
#     # loop over samples
#     output = []
#     for i, (f, h, w) in enumerate(grid_sizes.tolist()):
#         seq_len = f * h * w
#
#         # precompute multipliers
#         x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
#             s, n, -1, 2))
#         freqs_i = torch.cat([
#             freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
#             freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
#             freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
#         ],
#                             dim=-1).reshape(seq_len, 1, -1)
#
#         print(11111,freqs_i.shape)
#
#         # apply rotary embedding
#         # sp_size = get_sequence_parallel_world_size()
#         sp_size = nccl_info.sp_size
#         # sp_rank = get_sequence_parallel_rank()
#         sp_rank = nccl_info.rank_within_group
#         freqs_i = pad_freqs(freqs_i, s * sp_size)
#         print(22222,freqs_i.shape)
#         s_per_rank = s
#         freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *s_per_rank), :, :]
#         print(33333,freqs_i_rank.shape,x_i.shape)
#         x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
#         x_i = torch.cat([x_i, x[i, s:]])
#         print(44444,x_i.shape)
#         '''
#         11111 torch.Size([20280, 1, 64])
#         22222 torch.Size([20280, 1, 64])
#         33333 torch.Size([20280, 1, 64]) torch.Size([20280, 40, 64])
#         44444 torch.Size([20280, 40, 128])
#         '''
#         import os
#         os._exit(23333)
#         # append to collection
#         output.append(x_i)
#     return torch.stack(output).float()

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs,nccl_info):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        # q=rope_apply(q, grid_sizes, freqs,nccl_info)
        # k=rope_apply(k, grid_sizes, freqs,nccl_info)


        if nccl_info.sp_size>1:
            q=gather_sequence(q,dim=1,grad_scale="up")
            k=gather_sequence(k,dim=1,grad_scale="up")
        q=rope_apply(q, grid_sizes, freqs)
        k=rope_apply(k, grid_sizes, freqs)
        if nccl_info.sp_size>1:
            q=split_sequence(q,dim=1,grad_scale="down")
            k=split_sequence(k,dim=1,grad_scale="down")

        # print(444444444444,x.shape,q.shape,k.shape,v.shape,x.dtype,q.dtype,k.dtype,v.dtype)#32#32#32#bf16
        #torch.Size([1, 10140, 5120]) torch.Size([1, 10140, 40, 128]) torch.Size([1, 10140, 40, 128]) torch.Size([1, 10140, 40, 128])#BTHC
        if nccl_info.sp_size>1:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        if nccl_info.sp_size>1:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens,nccl_info):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens,nccl_info):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)
        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        #self.audio_attn=PerceiverAttentionCA(dim=dim,kv_dim=3072, dim_head=1024, heads=33)
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,nccl_info
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,freqs,nccl_info)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens,nccl_info)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()
        #self.audio_proj=AudioProjNet2(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=3072, context_tokens=4)        
        self.gradient_checkpointing = True
        self.training=False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,is_train=False,nccl_info=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        self.training=is_train
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # time embeddings
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            nccl_info=nccl_info,
            context_lens=context_lens)
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # Context Parallel
        if nccl_info.sp_size > 1:
            x = split_sequence(x, dim=1, grad_scale="down")
        # for block in self.blocks[:6]:
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # # head
        x = self.head(x, e)

        # Context Parallel
        if nccl_info.sp_size > 1:
            x = gather_sequence(x, dim=1, grad_scale="up")


        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

class Wan_Audio(ModelMixin, ConfigMixin):
    def __init__(self,wan_model,audio_proj):
        super().__init__()
        self.wan_model=wan_model
        self.audio_adapters=nn.ModuleList([AudioAttentionFA(dim=wan_model.dim,kv_dim=3072, dim_head=1024, heads=33) for _ in range(wan_model.num_layers)])
        self.audio_proj=audio_proj
        self.scale=1.0

    def forward(
        self,
        x,
        audio_fea,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,is_train=False,nccl_info=None,bbox=None
    ):
        self.wan_model.training=is_train
        if self.wan_model.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.wan_model.patch_embedding.weight.device

        if self.wan_model.freqs.device != device:
            self.wan_model.freqs = self.wan_model.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        #1*36*13*64*64
        # embeddingstorch.cat([x[0], y[0]], dim=0)
        try:
            x = [self.wan_model.patch_embedding(u.unsqueeze(0).to(device)) for u in x]
        except:
            import os,time
            if int(os.environ["LOCAL_RANK"]) <= 0:
                import pdb
                pdb.set_trace()
            else:
                #休眠
                time.sleep(100000)

        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        # time embeddings
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float32):
            e = self.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.wan_model.freq_dim, t).float())
            e0 = self.wan_model.time_projection(e).unflatten(1, (6, self.wan_model.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # context
        context_lens = None
        context = self.wan_model.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.wan_model.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = self.wan_model.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        audio_fea=self.audio_proj(audio_fea)
        
        audio_feature_pad = audio_fea[:,:1].repeat(1,3,1,1) 
        audio_fea = torch.cat([audio_feature_pad, audio_fea], dim=1).view(audio_fea.shape[0], 13, 16, 3072)#1*21*16*3072

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.wan_model.freqs,
            context=context,
            nccl_info=nccl_info,
            context_lens=context_lens)
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # Context Parallel
        if nccl_info.sp_size > 1:
            x = split_sequence(x, dim=1, grad_scale="down")
        bsz=x.shape[0]
        ot=13

        # for block in self.blocks[:6]:
        for idx,block in enumerate(self.wan_model.blocks):
            if self.wan_model.training and self.wan_model.gradient_checkpointing:
            #if False:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                #x=self.audio_adapters[idx](audio_fea,x.view(bsz, ot, -1, 5120)  ,self.scale).view(bsz,-1,5120)
                x = torch.utils.checkpoint.checkpoint(
                    self.audio_adapters[idx],
                    audio_fea,x.view(bsz, ot, -1, 5120)  ,self.scale,bbox
                ).view(x.shape[0],-1,5120)
                
            else:
                x = block(x, **kwargs)
                x=self.audio_adapters[idx](audio_fea,x.view(bsz, ot, -1, 5120),self.scale,bbox).view(x.shape[0],-1,5120)

        # # head
        x = self.wan_model.head(x, e)

        # Context Parallel
        if nccl_info.sp_size > 1:
            x = gather_sequence(x, dim=1, grad_scale="up")


        # unpatchify
        x = self.wan_model.unpatchify(x, grid_sizes)
        return [u.float() for u in x]