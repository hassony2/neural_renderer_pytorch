import chainer
import numpy as np
import torch


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndim == 3)
    assert (faces.ndim == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    xp = chainer.cuda.get_array_module(faces)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (xp.arange(bs, dtype='int32') * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces]


def vertices_to_faces_th(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.dim() == 3)
    assert (faces.dim() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces.int() + (
        (vertices.new(np.arange(bs)) * nv).int()).view(-1, 1, 1)
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces.long()]
