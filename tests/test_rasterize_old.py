import torch
import numpy as np

from neurender.rasterize import (get_line_coeffs, get_line_coeff, order_points,
                                 Rasterize)
from neurender.vertices_to_faces import vertices_to_faces, vertices_to_faces_th
from tests import utils

vertices, faces, textures = utils.load_teapot_batch()
vertices_th, faces_th, textures_th = utils.load_teapot_batch_th()


def test_load_teapot():
    assert vertices.shape == vertices_th.shape
    assert faces.shape == faces_th.shape
    assert textures.shape == textures_th.shape
    assert vertices.sum() - vertices_th.sum().item() == 0
    assert faces.sum() - faces_th.sum().item() == 0
    assert textures.sum() - textures_th.sum().item() == 0


faces = vertices_to_faces(vertices, faces)
faces_th = vertices_to_faces_th(vertices_th, faces_th)

image_size = 256
anti_aliasing = False


def test_vertices_to_faces():
    assert faces.shape == faces_th.shape
    assert np.abs(faces_th.sum().numpy() - faces.sum().get()) < 10e-5


def test_get_line_coeff():
    face = torch.Tensor([[1, 2], [1, 7], [5, 2]])
    a, b, c = get_line_coeff(face[0], face[1])
    assert a * face[0][0].item() + b * face[0][1].item() + c == 0
    assert a * face[1][0].item() + b * face[1][1].item() + c == 0
    a, b, c = get_line_coeff(face[1], face[2])
    assert a * face[1][0].item() + b * face[1][1].item() + c == 0
    assert a * face[2][0].item() + b * face[2][1].item() + c == 0
    a, b, c = get_line_coeff(face[0], face[2])
    assert a * face[0][0].item() + b * face[0][1].item() + c == 0
    assert a * face[2][0].item() + b * face[2][1].item() + c == 0


def test_order_points():
    face = torch.Tensor([[1, 2], [1, 7], [5, 2]])
