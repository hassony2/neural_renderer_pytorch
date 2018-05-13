import math

import pytest
import chainer
import chainer.functions as cf
import chainer.gradient_check
import chainer.testing
import cupy as cp
import torch
import scipy.misc
import numpy as np

from tests import utils
from neurender.renderer_th import Renderer as RendererTh
from neurender.renderer import Renderer
from neurender.perspective import perspective
from neurender.vertices_to_faces import vertices_to_faces
from neurender.rasterize import rasterize_silhouettes
from neurender.look_at import look_at
from neurender.perspective_th import perspective_th
from neurender.vertices_to_faces import vertices_to_faces_th
from neurender.look_at_th import look_at_th
from neurender.rasterize_th import rasterize_silhouettes as rasterize_silhouettes_th


def test_compare_preprocess():
    vertices, faces, textures = utils.load_teapot_batch()
    viewing_angle = 30
    eye = [0, 0, -(1. / math.tan(math.radians(viewing_angle)) + 1)]
    look_at_vertices = look_at(vertices, eye)
    perspective_vertices = perspective(look_at_vertices, angle=viewing_angle)
    faces_2 = vertices_to_faces(perspective_vertices, faces)

    vertices_th, faces_th, _ = utils.load_teapot_batch_th()
    eye = [0, 0, -(1. / math.tan(math.radians(viewing_angle)) + 1)]
    look_at_vertices_th = look_at_th(vertices_th, eye)
    perspective_vertices_th = perspective_th(
        look_at_vertices_th, angle=viewing_angle)
    faces_2_th = vertices_to_faces_th(perspective_vertices_th, faces_th)
    assert np.mean(np.abs(vertices.get() - vertices_th.numpy())) == 0
    assert np.mean(
        np.abs(look_at_vertices.data.get() -
               look_at_vertices_th.numpy())) < 1e-5
    assert np.mean(
        np.abs(perspective_vertices.data.get() -
               perspective_vertices_th.numpy())) < 1e-5
    assert np.mean(np.abs(faces_2.data.get() - faces_2_th.numpy())) < 1e-5


@pytest.mark.skip(reason="slow")
def test_forward_chainer():
    """Whether a silhouette by neural renderer matches that by Blender."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch()

    # create renderer
    renderer = Renderer()
    renderer.fill_back = False
    renderer.image_size = 256
    renderer.anti_aliasing = False

    images = renderer.render_silhouettes(vertices, faces)
    images = images.data.get()
    image = images[2]

    # load reference image by blender
    ref = scipy.misc.imread('./tests/data/teapot_blender.png')
    ref = ref.astype('float32')
    ref = (ref.min(-1) != 255).astype('float32')

    chainer.testing.assert_allclose(ref, image)


# @pytest.mark.skip(reason="slow")
def test_forward_th():
    """Whether a silhouette by neural renderer matches that by Blender."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch()
    vertices_th, faces_th, textures_th = utils.load_teapot_batch_th()

    # create renderer
    renderer_th = RendererTh()
    renderer_th.image_size = 256
    renderer_th.anti_aliasing = False

    renderer = Renderer()
    renderer.fill_back = False
    renderer.image_size = 256
    renderer.anti_aliasing = False

    images = renderer.render_silhouettes(vertices, faces)
    images_th = renderer.render_silhouettes(
        cp.asarray(vertices_th.cpu().numpy()),
        cp.asarray(faces_th.cpu().numpy()))
    assert (images_th - images).data.get().sum() == 0


#     images = images.data.get()
#     image = images[2]
#
#     # load reference image by blender
#     # ref = scipy.misc.imread('./tests/data/teapot_blender.png')
#     # ref = ref.astype('float32')
#     # ref = (ref.min(-1) != 255).astype('float32')
#
#     # chainer.testing.assert_allclose(ref, image)
