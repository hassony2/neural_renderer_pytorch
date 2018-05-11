import chainer
import chainer.gradient_check
import chainer.testing
import numpy as np
import torch

from neurender.look_at import look_at
from neurender.look_at_th import look_at_th


def test_chainer_look_at():
    eyes = [
        [1, 0, 1],
        [0, 0, -10],
        [-1, 1, 0],
    ]
    answers = [
        [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        [1, 0, 10],
        [0, np.sqrt(2) / 2, 3. / 2. * np.sqrt(2)],
    ]
    vertices = np.array([1, 0, 0], 'float32')
    vertices = vertices[None, None, :]
    for e, a in zip(eyes, answers):
        eye = np.array(e, 'float32')
        transformed = look_at(vertices, eye)
        chainer.testing.assert_allclose(transformed.data.flatten(),
                                        np.array(a))


def test_th_look_at():
    eyes = [
        [1, 0, 1],
        [0, 0, -10],
        [-1, 1, 0],
    ]
    answers = [
        [-np.sqrt(2) / 2, 0, np.sqrt(2) / 2],
        [1, 0, 10],
        [0, np.sqrt(2) / 2, 3. / 2. * np.sqrt(2)],
    ]
    vertices = np.array([1, 0, 0], 'float32')
    vertices = vertices[None, None, :]
    for e, a in zip(eyes, answers):
        eye = torch.Tensor(np.array(e, 'float32'))
        transformed = look_at_th(torch.Tensor(vertices), eye)
        assert (transformed - torch.Tensor(a)).norm().item() < 1e-6
