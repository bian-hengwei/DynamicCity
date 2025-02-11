import torch


def hexplane_to_rollout(hexplane):
    """
    Padded rollout, converts hexplane to (b, c, x+t+z, y+z+t)
    - xy: :x, :y
    - xz: :x, y:y+z
    - zy: x+t:,
    - tx: :x, y+z:
    - ty: x:x+t :y
    - tz: x:x+t y:y+z
    """
    xy, xz, yz, tx, ty, tz = hexplane
    B, C, T, X, Y, Z = *xy.shape[:-2], *tx.shape[2:], *yz.shape[2:]

    xy_xz_xt = torch.cat([xy, xz, tx.transpose(-1, -2)], dim=-1)  # B, C, X, Y+Z+T

    tt = torch.zeros(B, C, T, T, dtype=xy.dtype, device=xy.device)
    ty_tz_tt = torch.cat([ty, tz, tt], dim=-1)  # B, C, T, Y+Z+T

    zz = torch.zeros(B, C, Z, Z, dtype=xy.dtype, device=xy.device)
    zt = torch.zeros(B, C, Z, T, dtype=xy.dtype, device=xy.device)
    zy_zz_zt = torch.cat([yz.transpose(-1, -2), zz, zt], dim=-1)  # B, C, Z, Y+Z+T

    return torch.cat([xy_xz_xt, ty_tz_tt, zy_zz_zt], dim=-2)  # B, C, X+T+Z, Y+Z+T


def rollout_to_hexplane(rollout, txyz):
    T, X, Y, Z = txyz
    xy = rollout[..., :X, :Y]  # B, C, X, Y
    xz = rollout[..., :X, Y: Y + Z]  # B, C, X, Z
    yz = rollout[..., X + T:, :Y].transpose(-1, -2)  # B, C, Y, Z
    tx = rollout[..., :X, Y + Z:].transpose(-1, -2)  # B, C, T, X
    ty = rollout[..., X: X + T, :Y]  # B, C, T, Y
    tz = rollout[..., X: X + T, Y: Y + Z]  # B, C, T, Z
    return xy, xz, yz, tx, ty, tz


def get_rollout_mask(txyz):
    T, X, Y, Z = txyz
    hexplane_mask = (
        torch.ones(1, 1, X, Y),
        torch.ones(1, 1, X, Z),
        torch.ones(1, 1, Y, Z),
        torch.ones(1, 1, T, X),
        torch.ones(1, 1, T, Y),
        torch.ones(1, 1, T, Z),
    )
    return hexplane_to_rollout(hexplane_mask)
