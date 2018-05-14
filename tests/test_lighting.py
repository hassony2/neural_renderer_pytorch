import numpy as np
import torch

from neurender.lighting import lighting, lighting_th


def test_lighting():
    """Test whether it is executable."""
    faces = np.random.normal(size=(64, 16, 3, 3)).astype('float32')
    textures = np.random.normal(size=(64, 16, 8, 8, 8, 3)).astype('float32')
    lighted_textures = lighting(faces, textures)

    textures_th = torch.Tensor(textures)
    faces_th = torch.Tensor(faces)
    lighted_textures_th = lighting_th(faces_th, textures_th)
    mean_err = np.mean(
        np.abs(lighted_textures.data - lighted_textures_th.numpy()))
    assert mean_err < 1e-5
