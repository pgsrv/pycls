#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RegNet models."""

import numpy as np
from pycls.core.config import cfg
from pycls.models.anynet import AnyNet


def quantize_float(f, q):
    """Converts a float to closest int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bs, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bs)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bs)]
    return ws, gs


def get_stages_from_blocks(ws_block):
    """Gets ws/ds of network at each stage from per block values."""
    ts = [w != w_p for w, w_p in zip(ws_block + [0], [0] + ws_block)]
    ws = [w for w, t in zip(ws_block, ts[:-1]) if t]
    ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return ws, ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_block = w_0 * np.power(w_m, ks)
    ws_block = np.round(np.divide(ws_block, q)) * q
    num_stages, max_stage = len(np.unique(ws_block)), ks.max() + 1
    ws_block, ws_cont = ws_block.astype(int).tolist(), ws_cont.tolist()
    return ws_block, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def get_params():
        """Convert RegNet to AnyNet parameter format."""
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
        ws_block, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(ws_block)
        # Use the same g, b and s for each stage
        gs = [cfg.REGNET.GROUP_W for _ in range(num_stages)]
        bs = [cfg.REGNET.BOT_MUL for _ in range(num_stages)]
        ss = [cfg.REGNET.STRIDE for _ in range(num_stages)]
        # Adjust the compatibility of ws and gws
        ws, gs = adjust_ws_gs_comp(ws, bs, gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.REGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self):
        params = RegNet.get_params()
        super(RegNet, self).__init__(params)

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        params = RegNet.get_params() if not params else params
        return AnyNet.complexity(cx, params)
