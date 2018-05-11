import chainer
import chainer.functions as cf
import torch


def perspective_th(vertices, angle=30.):
    assert (vertices.ndim == 3)
    angle = torch.Tensor([angle])
    angle = angle / 180. * 3.1416
    angle.repeat(vertices.shape[0])

    width = torch.tan(angle)
    width = width.unsqueeze(1).repeat(vertices.shape[:2])
    z = torch.Tensor(vertices[:, :, 2])
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack([x, y, z], dim=2)
    return vertices
