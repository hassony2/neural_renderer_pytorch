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
from neurender.rasterize_th import RasterizeSil


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


def test_compare_preprocess_teapot():
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


def test_compare_preprocess_simple():
    # Prepare chainer arrays
    viewing_angle = 30
    eye = [0, 0, -(1. / math.tan(math.radians(viewing_angle)) + 1)]
    vertices = np.array([[0.8, 0.8, 1.], [0.0, -0.5, 1.], [0.2, -0.4, 1.]])
    faces = np.array([[0, 1, 2]])

    vertices_ch = cp.array(vertices, 'float32')
    faces_ch = cp.array(faces, 'int32')
    vertices_ch, faces_ch = utils.to_minibatch((vertices_ch, faces_ch))

    # Prepare torch arrays
    vertices_th, faces_th = utils.to_minibatch_th((vertices, faces))

    look_at_vertices = look_at(vertices_ch, eye)
    perspective_vertices = perspective(look_at_vertices, angle=viewing_angle)
    faces_2 = vertices_to_faces(perspective_vertices, faces_ch)

    look_at_vertices_th = look_at_th(vertices_th, eye)
    perspective_vertices_th = perspective_th(
        look_at_vertices_th, angle=viewing_angle)
    faces_2_th = vertices_to_faces_th(perspective_vertices_th, faces_th)
    assert np.mean(np.abs(vertices_ch.get() - vertices_th.numpy())) == 0
    assert np.mean(
        np.abs(look_at_vertices.data.get() -
               look_at_vertices_th.numpy())) < 1e-5
    assert np.mean(
        np.abs(perspective_vertices.data.get() -
               perspective_vertices_th.numpy())) < 1e-5
    assert np.mean(np.abs(faces_2.data.get() - faces_2_th.numpy())) < 1e-5


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


def test_backward_silhouette():
    """Backward if non-zero gradient is out of a face."""

    grad_ref = [
        [1.6725862, -0.26021874, 0.],
        [1.41986704, -1.64284933, 0.],
        [0., 0., 0.],
    ]
    vertices = [[0.8, 0.8, 1.], [0.0, -0.5, 1.], [0.2, -0.4, 1.]]
    faces = [[0, 1, 2]]

    vertices = cp.array(vertices, 'float32')
    faces = cp.array(faces, 'int32')
    grad_ref = cp.array(grad_ref, 'float32')
    vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
    pxi = 35
    pyi = 25

    renderer = Renderer()
    renderer.image_size = 64
    renderer.anti_aliasing = False
    renderer.fill_back = False
    renderer.perspective = False
    print(vertices.shape)
    print(faces.shape)
    vertices = chainer.Variable(vertices)
    images = renderer.render_silhouettes(vertices, faces)
    loss = cf.sum(cf.absolute(images[:, pyi, pxi] - 1))
    loss.backward()
    chainer.testing.assert_allclose(vertices.grad, grad_ref, rtol=1e-2)


def test_backward_silhouette_th():
    """Backward if non-zero gradient is out of a face."""

    vertices = np.array([[0.8, 0.8, 1.], [0.0, -0.5, 1.], [0.2, -0.4, 1.]])
    faces = np.array([[0, 1, 2]]).astype('int32')
    grad_ref = np.array([
        [1.6725862, -0.26021874, 0.],
        [1.41986704, -1.64284933, 0.],
        [0., 0., 0.],
    ])
    vertices, faces, grad_ref = utils.to_minibatch_th((vertices, faces,
                                                       grad_ref))
    print(vertices.shape, faces.shape)

    faces.requires_grad = True
    vertices.requires_grad = True
    pxi = 35
    pyi = 25
    faces_th = preprocess_th(vertices, faces, perspective=False)
    rasterize_silhouettes_th = RasterizeSil(64, 0.1, 100, 1e-3, [0, 0, 0])
    images = rasterize_silhouettes_th(faces_th)
    loss = torch.sum(torch.abs(images[:, pyi, pxi] - 1))
    loss.backward(retain_graph=True)
    assert (vertices.grad - grad_ref).abs().mean() < 1e-3


def test_backward_silhouette_ch_2():
    """Backward if non-zero gradient is on a face."""

    vertices = np.array([[0.8, 0.8, 1.], [-0.5, -0.8, 1.], [0.8, -0.8, 1.]])
    faces = np.array([[0, 1, 2]])
    pyi = 40
    pxi = 50
    grad_ref = np.array([
        [0.98646867, 1.04628897, 0.],
        [-1.03415668, -0.10403691, 0.],
        [3.00094461, -1.55173182, 0.],
    ])

    renderer = Renderer()
    renderer.image_size = 64
    renderer.anti_aliasing = False
    renderer.perspective = False

    # Prepare chainer inputs
    vertices = cp.array(vertices, 'float32')
    faces = cp.array(faces, 'int32')
    grad_ref = cp.array(grad_ref, 'float32')
    vertices, faces, grad_ref = utils.to_minibatch((vertices, faces, grad_ref))
    vertices = chainer.Variable(vertices)
    images = renderer.render_silhouettes(vertices, faces)
    loss = cf.sum(cf.absolute(images[:, pyi, pxi]))
    loss.backward()

    chainer.testing.assert_allclose(vertices.grad, grad_ref, rtol=1e-2)


def test_backward_silhouette_th_2():
    """Backward if non-zero gradient is on a face."""

    vertices = np.array([[0.8, 0.8, 1.], [-0.5, -0.8, 1.], [0.8, -0.8, 1.]])
    faces = np.array([[0, 1, 2]])
    pyi = 40
    pxi = 50
    grad_ref = np.array([
        [0.98646867, 1.04628897, 0.],
        [-1.03415668, -0.10403691, 0.],
        [3.00094461, -1.55173182, 0.],
    ])
    vertices, faces, grad_ref = utils.to_minibatch_th((vertices, faces,
                                                       grad_ref))
    faces.requires_grad = True
    vertices.requires_grad = True
    faces_th = preprocess_th(vertices, faces, perspective=False)
    rasterize_silhouettes_th = RasterizeSil(64, 0.1, 100, 1e-3, [0, 0, 0])
    images = rasterize_silhouettes_th(faces_th)
    loss = torch.sum(torch.abs(images[:, pyi, pxi]))
    loss.backward(retain_graph=True)
    assert (vertices.grad - grad_ref).abs().mean() < 1e-3
