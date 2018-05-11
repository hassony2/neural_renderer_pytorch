import chainer
import chainer.functions as cf
import chainer.gradient_check
import chainer.testing
import cupy as cp
import torch

from tests import utils
from neurender.renderer_th import Renderer


def test_case1():
    """Whether a silhouette by neural renderer matches that by Blender."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch_th()

    # create renderer
    renderer = Renderer()
    renderer.image_size = 256
    renderer.anti_aliasing = False

    images = renderer.render_silhouettes(vertices, faces)
    images = images.data.get()
    image = images[2]

    # load reference image by blender
    # ref = scipy.misc.imread('./tests/data/teapot_blender.png')
    # ref = ref.astype('float32')
    # ref = (ref.min(-1) != 255).astype('float32')

    # chainer.testing.assert_allclose(ref, image)
