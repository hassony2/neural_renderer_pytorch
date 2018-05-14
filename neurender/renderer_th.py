import math

import chainer.functions as cf
import torch

from neurender.perspective_th import perspective_th
from neurender.vertices_to_faces import vertices_to_faces_th
from neurender.look_at_th import look_at_th
from neurender.lighting import lighting_th
from neurender.rasterize import rasterize_silhouettes, rasterize
from neurender.rasterize_th import RasterizeRGB, RasterizeSil


class Renderer(object):
    def __init__(self):
        # rendering
        self.image_size = 256
        self.anti_aliasing = True
        self.background_color = [0, 0, 0]
        self.fill_back = True

        # camera
        self.perspective = True
        self.viewing_angle = 30
        self.eye = [
            0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)
        ]
        self.camera_mode = 'look_at'
        self.camera_direction = [0, 0, 1]
        self.near = 0.1
        self.far = 100

        # light
        self.light_intensity_ambient = 0.5
        self.light_intensity_directional = 0.5
        self.light_color_ambient = [1, 1, 1]  # white
        self.light_color_directional = [1, 1, 1]  # white
        self.light_direction = [0, 1, 0]  # up-to-down

        # rasterization
        self.rasterizer_eps = 1e-3
        self.rasterize_rgb = RasterizeRGB(self.image_size, self.near, self.far,
                                          self.rasterizer_eps,
                                          self.background_color)
        self.rasterize_sil = RasterizeSil(self.image_size, self.near, self.far,
                                          self.rasterizer_eps,
                                          self.background_color)

    def render_silhouettes(self, vertices, faces):
        if self.fill_back:
            faces = torch.cat(
                [faces, faces[:, :, faces.new([2, 1, 0]).long()]], dim=1)
        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = look_at_th(vertices, self.eye)
        else:
            raise NotImplementedError

        # perspective transformation
        if self.perspective:
            vertices = perspective_th(vertices, angle=self.viewing_angle)

        # rasterization
        faces = vertices_to_faces_th(vertices, faces)
        images = self.rasterize_sil(faces)
        return images

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = torch.cat(
                [faces, faces[:, :, faces.new([2, 1, 0]).long()]], dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = look_at_th(vertices, self.eye)
        elif self.camera_mode == 'look':
            raise NotImplementedError

        # perspective transformation
        if self.perspective:
            vertices = perspective_th(vertices, angle=self.viewing_angle)

        # rasterization
        faces = vertices_to_faces_th(vertices, faces)
        images = rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = torch.cat(
                [faces, faces[:, :, faces.new([2, 1, 0]).long()]], dim=1)
            textures = torch.cat(
                [textures, textures.permute(0, 1, 4, 3, 2, 5)], dim=1)

        # lighting
        faces_lighting = vertices_to_faces_th(vertices, faces)
        textures = lighting_th(
            faces_lighting, textures, self.light_intensity_ambient,
            self.light_intensity_directional, self.light_color_ambient,
            self.light_color_directional, self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = look_at_th(vertices, self.eye)
        elif self.camera_mode == 'look':
            raise NotImplementedError
            vertices = look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = perspective_th(vertices, angle=self.viewing_angle)

        # rasterization
        faces = vertices_to_faces_th(vertices, faces)
        images = self.rasterize_rgb(faces, textures)
        return images
