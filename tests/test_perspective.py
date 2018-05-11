import chainer
import chainer.gradient_check
import chainer.testing
import numpy as np
import torch

from neurender.perspective import perspective
from neurender.perspective_th import perspective_th


def test_perspective():
    v_in = [1, 2, 10]
    v_out = [np.sqrt(3) / 10, 2 * np.sqrt(3) / 10, 10]
    vertices = np.array(v_in, 'float32')
    vertices = vertices[None, None, :]
    transformed = perspective(vertices)
    chainer.testing.assert_allclose(transformed.data.flatten(),
                                    np.array(v_out, 'float32'))


def test_perspective_th():
    v_in = [1, 2, 10]
    v_out = [np.sqrt(3) / 10, 2 * np.sqrt(3) / 10, 10]
    vertices = np.array(v_in, 'float32')
    vertices = vertices[None, None, :]
    transformed = perspective_th(vertices)
    assert (torch.Tensor(v_out) - transformed).norm().item() < 1e-5
