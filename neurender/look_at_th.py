import chainer
import chainer.functions as cf
import numpy as np
import torch
import neurender


def look_at_th(vertices, eye, at=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndim == 3)

    batch_size = vertices.shape[0]
    if at is None:
        at = np.array([0, 0, 0], 'float32')
    if up is None:
        up = np.array([0, 1, 0], 'float32')

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = np.array(eye, 'float32')
    eye = torch.Tensor(eye)
    at = torch.Tensor(at)
    up = torch.Tensor(up)
    if eye.dim() == 1:
        eye = eye.unsqueeze(0).repeat(batch_size, 1)
    if at.dim() == 1:
        at = at.unsqueeze(0).repeat(batch_size, 1)
    if up.dim() == 1:
        up = up.unsqueeze(0).repeat(batch_size, 1)

    # create new axes
    z_axis_norm = (at - eye) / (at - eye).norm()
    x_axis = torch.cross(up, z_axis_norm)
    x_axis_norm = x_axis / x_axis.norm()
    y_axis = torch.cross(z_axis_norm, x_axis)
    y_axis_norm = y_axis / y_axis.norm()

    # create rotation matrix: [bs, 3, 3]
    rot = torch.cat([x_axis_norm, y_axis_norm,
                     z_axis_norm]).unsqueeze(0).repeat(vertices.shape[0], 1, 1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye.unsqueeze(0).repeat(vertices.shape[0], vertices.shape[1], 1)
    vertices = vertices - eye
    vertices = torch.matmul(vertices, rot.transpose(1, 2))

    return vertices
