# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class EdgeEnhancedPositionEmbedding(nn.Module):
    """
    Combines standard sinusoidal position embedding with learnable edge information
    extracted from the input feature map.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        # Following settings only relevant
        # for warmping up cache for compilation
        warmup_cache: bool = True,
        image_size: int = 1024,
        strides: Tuple[int, int, int, int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        # --- Learnable Edge Detection Kernels (Initialized with Sobel) ---
        kernel_tl = torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        kernel_tr = torch.tensor([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=torch.float32)
        kernel_bl = torch.tensor([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=torch.float32)
        kernel_br = torch.tensor([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype=torch.float32)
        initial_kernels = torch.stack([kernel_tl, kernel_tr, kernel_bl, kernel_br]).unsqueeze(1)
        self.learnable_edge_kernels = nn.Parameter(initial_kernels)

        # --- Learnable scalar for residual edge injection ---
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # --- Projection layers for different feature scales ---
        self.edge_projection = nn.ModuleDict()

        self.cache = {}
        if warmup_cache and torch.cuda.is_available():
            device = torch.device("cuda")
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, device, *cache_key)

    def _get_edge_projection(self, in_channels, device):
        """
        Dynamically create and cache a projection layer for each feature scale
        to avoid re-initializing weights on every forward pass.
        """
        key = str(in_channels)
        if key not in self.edge_projection:
            # Projection layer for position encoding enhancement
            self.edge_projection[key] = nn.Conv2d(
                in_channels * 4, self.num_pos_feats * 2, kernel_size=1
            ).to(device)
        return self.edge_projection[key]

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device)

        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed / (y_embed[-1] + eps)) * self.scale
            x_embed = (x_embed / (x_embed[-1] + eps)) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)

        # Expand 1D embeddings to 2D of shape [H, W, C]
        pos_y = pos_y.unsqueeze(1).repeat(1, W, 1)
        pos_x = pos_x.unsqueeze(0).repeat(H, 1, 1)
        
        # Concatenate and permute to shape [2*C, H, W]
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)

        self.cache[cache_key] = pos
        return pos[None].repeat(B, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1. Get standard sinusoidal position embedding
        pos = self._pe(B, x.device, *(H, W))
        
        # 2. Extract edge features using learnable kernels
        # The kernels are repeated for each input channel, maintaining channel independence
        learnable_kernels_gpu = self.learnable_edge_kernels.repeat(C, 1, 1, 1).to(x.device)
        edge_conv = F.conv2d(x, learnable_kernels_gpu, padding=1, groups=C)
        
        # 3. Get scale-specific projection layer
        proj_layer = self._get_edge_projection(C, x.device)

        # 4. Generate projected edges for position encoding enhancement
        projected_edges = proj_layer(edge_conv)
        
        # 5. Inject edge information into the position embedding with learnable weight
        final_pos = pos + self.alpha * projected_edges

        return final_pos


# For backward compatibility with trainer.py which expects PositionEmbeddingSine
PositionEmbeddingSine = EdgeEnhancedPositionEmbedding