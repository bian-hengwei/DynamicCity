import torch
import torch.nn.functional as F


def get_pred_label(pred, dim=-1):
    pred = torch.softmax(pred, dim=dim)
    pred = pred.argmax(dim=dim)
    return pred


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def compose_hexplane_channelwise(feat_maps):
    h_xy, h_xz, h_yz, h_tx, h_ty, h_tz = feat_maps
    sizes_xyz = *_, X, Y, Z = *h_xy.shape[:2], h_tx.shape[2], *h_xy.shape[2:], h_xz.shape[3]

    w_new, h_new = max(X, Y), max(Y, Z)
    h_xy = F.pad(h_xy, (0, h_new - Y, 0, w_new - X))
    h_xz = F.pad(h_xz, (0, h_new - Z, 0, w_new - X))
    h_yz = F.pad(h_yz, (0, h_new - Z, 0, w_new - Y))
    h = torch.cat([h_xy, h_xz, h_yz], dim=1)

    sizes_t = *_, X, Y, Z = *h_tx.shape[:2], h_tx.shape[2], h_tx.shape[3], h_ty.shape[3], h_tz.shape[3]
    t_new = max(X, Y, Z)
    h_tx = F.pad(h_tx, (0, t_new - X))
    h_ty = F.pad(h_ty, (0, t_new - Y))
    h_tz = F.pad(h_tz, (0, t_new - Z))
    t = torch.cat([h_tx, h_ty, h_tz], dim=1)
    return h, t, sizes_xyz, sizes_t


def decompose_hexplane_channelwise(h, t, sizes_xyz, sizes_t):
    *_, X, Y, Z = sizes_xyz
    C = h.shape[1] // 3
    h_xy = h[:, : C, : X, : Y]
    h_xz = h[:, C: 2 * C, : X, : Z]
    h_yz = h[:, 2 * C:, : Y, : Z]

    *_, X, Y, Z = sizes_t
    C = t.shape[1] // 3
    h_tx = t[:, : C, :, : X]
    h_ty = t[:, C: 2 * C, :, : Y]
    h_tz = t[:, 2 * C:, :, : Z]
    return [h_xy, h_xz, h_yz, h_tx, h_ty, h_tz]


def add_positional_encoding(voxel, pos_num_freq):
    B, T, X, Y, Z, C = voxel.shape
    t = torch.linspace(0, T - 1, T)
    x = torch.linspace(0, X - 1, X)
    y = torch.linspace(0, Y - 1, Y)
    z = torch.linspace(0, Z - 1, Z)
    tt, xx, yy, zz = torch.meshgrid(t, x, y, z, indexing='ij')  # T, X, Y, Z
    coords = torch.stack([tt, xx, yy, zz], dim=-1).to(voxel.device)  # T, X, Y, Z, 4
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1)  # B, T, X, Y, Z, 4

    positional_encoding = list()
    for freq in range(pos_num_freq):
        positional_encoding.append(torch.sin((2. ** freq) * coords))
        positional_encoding.append(torch.cos((2. ** freq) * coords))
    positional_encoding = torch.cat(positional_encoding, dim=-1)  # B, T, X, Y, Z, 4 * 2 * pos_num_freq)

    voxel = torch.cat([voxel, positional_encoding], dim=-1)
    return voxel
