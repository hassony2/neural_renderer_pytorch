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
from neurender.lighting import lighting, lighting_th
from neurender.rasterize_th import RasterizeSil, RasterizeRGB


def preprocess_th(vertices_th, faces_th, viewing_angle=30, perspective=True):
    eye = [0, 0, -(1. / math.tan(math.radians(viewing_angle)) + 1)]
    look_at_vertices_th = look_at_th(vertices_th, eye)
    if perspective:
        perspective_vertices_th = perspective_th(
            look_at_vertices_th, angle=viewing_angle)
    else:
        perspective_vertices_th = look_at_vertices_th
    faces_th = vertices_to_faces_th(perspective_vertices_th, faces_th)
    return faces_th


def test_forward_ch_1():
    """Rendering a teapot without anti-aliasing."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch()

    # create renderer
    renderer = Renderer()
    renderer.image_size = 256
    renderer.anti_aliasing = False

    # render
    images = renderer.render(vertices, faces, textures)
    images = images.data.get()
    image = images[2]
    image = image.transpose((1, 2, 0))

    scipy.misc.imsave('./tests/data/test_rasterize1.png', image)


def test_forward_th_1():
    """Rendering a teapot without anti-aliasing."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch_th()

    # create renderer
    renderer = Renderer()
    renderer.image_size = 256
    renderer.anti_aliasing = False

    # render
    images_th = renderer.render(
        cp.asarray(vertices.cpu().numpy()),
        cp.asarray(faces.cpu().numpy()), cp.asarray(textures.cpu().numpy()))
    images = images_th.data.get()
    image = images[2]
    image = image.transpose((1, 2, 0))

    scipy.misc.imsave('./tests/data/test_rasterize_th1.png', image)
    ref = scipy.misc.imread('./tests/data/test_rasterize1.png')
    ref = ref.astype('float32')
    assert np.abs((ref / 255 - image)).mean() < 1e-3


def test_forward_th_2():
    """Rendering a teapot without anti-aliasing."""

    # load teapot
    vertices, faces, textures = utils.load_teapot_batch_th()

    # create renderer
    renderer = Renderer()
    renderer.image_size = 256
    renderer.anti_aliasing = False

    # render
    images_th = renderer.render(
        cp.asarray(vertices.cpu().numpy()),
        cp.asarray(faces.cpu().numpy()), cp.asarray(textures.cpu().numpy()))
    images = images_th.data.get()
    image = images[2]
    image = image.transpose((1, 2, 0))

    scipy.misc.imsave('./tests/data/test_rasterize_th1.png', image)
    ref = scipy.misc.imread('./tests/data/test_rasterize1.png')
    ref = ref.astype('float32')
    assert np.abs((ref / 255 - image)).mean() < 1e-3


def test_forward_rgb_th():
    """Backward if non-zero gradient is out of a face."""
    # load teapot
    vertices, faces, textures = utils.load_teapot_batch_th()

    # Add lighting
    light_intensity_ambient = 0.5
    light_intensity_directional = 0.5
    light_color_ambient = [1, 1, 1]  # white
    light_color_directional = [1, 1, 1]  # white
    light_direction = [0, 1, 0]  # up-to-down
    faces_lighting = vertices_to_faces_th(vertices, faces)
    textures = lighting_th(faces_lighting, textures, light_intensity_ambient,
                           light_intensity_directional, light_color_ambient,
                           light_color_directional, light_direction)
    faces_th = preprocess_th(vertices, faces, perspective=False)
    rasterize_rgb_th = RasterizeRGB(256, 0.1, 100, 1e-3, [0, 0, 0])
    images = rasterize_rgb_th(faces_th, textures)
    image = images[2].cpu().numpy()
    image = image.transpose((1, 2, 0))
    scipy.misc.imsave('./tests/data/test_rasterize_rgb_th1.png', image)


# def test_backward_silhouette():
#     """Backward if non-zero gradient is out of a face."""
#
#     grad_ref = [
#         [1.6725862, -0.26021874, 0.],
#         [1.41986704, -1.64284933, 0.],
#         [0., 0., 0.],
#     ]
#     vertices = [[0.8, 0.8, 1.], [0.0, -0.5, 1.], [0.2, -0.4, 1.]]
#     faces = [[0, 1, 2]]
#
#     vertices = cp.array(vertices, 'float32')
#     faces = cp.array(faces, 'int32')
#     grad_ref = cp.array(grad_ref, 'float32')
#     vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
#     pxi = 35
#     pyi = 25
#
#     renderer = Renderer()
#     renderer.image_size = 64
#     renderer.anti_aliasing = False
#     renderer.fill_back = False
#     renderer.perspective = False
#     print(vertices.shape)
#     print(faces.shape)
#     vertices = chainer.Variable(vertices)
#     images = renderer.render_silhouettes(vertices, faces)
#     loss = cf.sum(cf.absolute(images[:, pyi, pxi] - 1))
#     loss.backward()
#     chainer.testing.assert_allclose(vertices.grad, grad_ref, rtol=1e-2)
#
#
# def test_backward_silhouette_th():
#     """Backward if non-zero gradient is out of a face."""
#
#     vertices = np.array([[0.8, 0.8, 1.], [0.0, -0.5, 1.], [0.2, -0.4, 1.]])
#     faces = np.array([[0, 1, 2]]).astype('int32')
#     grad_ref = np.array([
#         [1.6725862, -0.26021874, 0.],
#         [1.41986704, -1.64284933, 0.],
#         [0., 0., 0.],
#     ])
#     vertices, faces, grad_ref = utils.to_minibatch_th((vertices, faces,
#                                                        grad_ref))
#     print(vertices.shape, faces.shape)
#
#     faces.requires_grad = True
#     vertices.requires_grad = True
#     pxi = 35
#     pyi = 25
#     faces_th = preprocess_th(vertices, faces, perspective=False)
#     rasterize_silhouettes_th = RasterizeSil(64, 0.1, 100, 1e-3, [0, 0, 0])
#     images = rasterize_silhouettes_th(faces_th)
#     loss = torch.sum(torch.abs(images[:, pyi, pxi] - 1))
#     loss.backward(retain_graph=True)
#     assert (vertices.grad - grad_ref).abs().mean() < 1e-3
#
#
# def test_backward_silhouette_ch_2():
#     """Backward if non-zero gradient is on a face."""
#
#     vertices = np.array([[0.8, 0.8, 1.], [-0.5, -0.8, 1.], [0.8, -0.8, 1.]])
#     faces = np.array([[0, 1, 2]])
#     pyi = 40
#     pxi = 50
#     grad_ref = np.array([
#         [0.98646867, 1.04628897, 0.],
#         [-1.03415668, -0.10403691, 0.],
#         [3.00094461, -1.55173182, 0.],
#     ])
#
#     renderer = Renderer()
#     renderer.image_size = 64
#     renderer.anti_aliasing = False
#     renderer.perspective = False
#
#     # Prepare chainer inputs
#     vertices = cp.array(vertices, 'float32')
#     faces = cp.array(faces, 'int32')
#     grad_ref = cp.array(grad_ref, 'float32')
#     vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
#     vertices = chainer.Variable(vertices)
#     images = renderer.render_silhouettes(vertices, faces)
#     loss = cf.sum(cf.absolute(images[:, pyi, pxi]))
#     loss.backward()
#
#     chainer.testing.assert_allclose(vertices.grad, grad_ref, rtol=1e-2)
#
#
# def test_backward_silhouette_th_2():
#     """Backward if non-zero gradient is on a face."""
#
#     vertices = np.array([[0.8, 0.8, 1.], [-0.5, -0.8, 1.], [0.8, -0.8, 1.]])
#     faces = np.array([[0, 1, 2]])
#     pyi = 40
#     pxi = 50
#     grad_ref = np.array([
#         [0.98646867, 1.04628897, 0.],
#         [-1.03415668, -0.10403691, 0.],
#         [3.00094461, -1.55173182, 0.],
#     ])
#     vertices, faces, grad_ref = utils.to_minibatch_th((vertices, faces,
#                                                        grad_ref))
#     faces.requires_grad = True
#     vertices.requires_grad = True
#     faces_th = preprocess_th(vertices, faces, perspective=False)
#     rasterize_silhouettes_th = RasterizeSil(64, 0.1, 100, 1e-3, [0, 0, 0])
#     images = rasterize_silhouettes_th(faces_th)
#     loss = torch.sum(torch.abs(images[:, pyi, pxi]))
#     loss.backward(retain_graph=True)
#     assert (vertices.grad - grad_ref).abs().mean() < 1e-3
