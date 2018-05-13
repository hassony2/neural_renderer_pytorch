"""
MIT License

Copyright (c) 2017 Hiroharu Kato

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import chainer
import numpy as np
import torch

from neurender import load_obj


def to_minibatch(data, batch_size=4, target_num=2):
    ret = []
    for d in data:
        xp = chainer.cuda.get_array_module(d)
        d2 = xp.repeat(xp.expand_dims(xp.zeros_like(d), 0), batch_size, axis=0)
        d2[target_num] = d
        ret.append(d2)
    return ret


def to_minibatch_th(data, batch_size=4, target_num=2):
    """
    Creates minibatch of zeros expept in idx target_num where it stores data
    """
    ret = []
    for d in data:
        d_batch = torch.Tensor(np.zeros_like(d)).unsqueeze(0).repeat(
            batch_size, * [1] * d.ndim)
        d_batch[target_num] = torch.Tensor(d)
        ret.append(d_batch)
    return ret


def load_teapot_batch(batch_size=4, target_num=2):
    vertices, faces = load_obj.load_obj('tests/data/teapot.obj')

    textures = np.ones((faces.shape[0], 4, 4, 4, 3), 'float32')
    vertices, faces, textures = to_minibatch((vertices, faces, textures),
                                             batch_size, target_num)
    vertices = chainer.cuda.to_gpu(vertices)
    faces = chainer.cuda.to_gpu(faces)
    textures = chainer.cuda.to_gpu(textures)
    return vertices, faces, textures


def load_teapot_batch_th(batch_size=4, target_num=2):
    vertices, faces = load_obj.load_obj('tests/data/teapot.obj')

    textures = np.ones((faces.shape[0], 4, 4, 4, 3), 'float32')
    vertices_th, faces_th, textures_th = to_minibatch_th(
        (vertices, faces, textures), batch_size, target_num)
    return vertices_th, faces_th.int(), textures_th
