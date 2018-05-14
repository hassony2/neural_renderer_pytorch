import chainer
import chainer.functions as cf
import chainer.functions.math.basic_math as cfmath
import torch

from neurender.cross import cross


def lighting(faces,
             textures,
             intensity_ambient=0.5,
             intensity_directional=0.5,
             color_ambient=(1, 1, 1),
             color_directional=(1, 1, 1),
             direction=(0, 1, 0)):
    xp = chainer.cuda.get_array_module(faces)
    bs, nf = faces.shape[:2]

    # arguments
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = xp.array(color_ambient, 'float32')
    if isinstance(color_directional, tuple) or isinstance(
            color_directional, list):
        color_directional = xp.array(color_directional, 'float32')
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = xp.array(direction, 'float32')
    if color_ambient.ndim == 1:
        color_ambient = cf.broadcast_to(color_ambient[None, :], (bs, 3))
    if color_directional.ndim == 1:
        color_directional = cf.broadcast_to(color_directional[None, :], (bs,
                                                                         3))
        if direction.ndim == 1:
            direction = cf.broadcast_to(direction[None, :], (bs, 3))

    # create light
    light = xp.zeros((bs, nf, 3), 'float32')

    # ambient light
    if intensity_ambient != 0:
        light = light + intensity_ambient * cf.broadcast_to(
            color_ambient[:, None, :], light.shape)

        # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        normals = cf.normalize(cross(v10, v12))
        normals = normals.reshape((bs, nf, 3))

        if direction.ndim == 2:
            direction = cf.broadcast_to(direction[:, None, :], normals.shape)
        cos = cf.relu(cf.sum(normals * direction, axis=2))
        light = (light + intensity_directional * cfmath.mul(
            *cf.broadcast(color_directional[:, None, :], cos[:, :, None])))

        # apply
    light = cf.broadcast_to(light[:, :, None, None, None, :], textures.shape)
    textures = textures * light
    return textures


def lighting_th(faces,
                textures,
                intensity_ambient=0.5,
                intensity_directional=0.5,
                color_ambient=(1, 1, 1),
                color_directional=(1, 1, 1),
                direction=(0, 1, 0)):
    bs, nf = faces.shape[:2]

    # arguments
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = faces.new(color_ambient).float()
    if isinstance(color_directional, tuple) or isinstance(
            color_directional, list):
        color_directional = faces.new(color_directional).float()
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = faces.new(direction).float()
    if color_ambient.dim() == 1:
        color_ambient = color_ambient.unsqueeze(0).repeat(bs, 1)
    if color_directional.dim() == 1:
        color_directional = color_directional.unsqueeze(0).repeat(bs, 1)
    if direction.dim() == 1:
        direction = direction.unsqueeze(0).repeat(bs, 1)

    # create light
    light = torch.zeros(bs, nf, 3)

    # ambient light
    if intensity_ambient != 0:
        light = light + intensity_ambient * color_ambient.unsqueeze(1)

    # directional light
    if intensity_directional != 0:
        faces = faces.view((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        normals = torch.cross(v10, v12)
        normals_norm = normals / torch.norm(normals, p=2, dim=1).unsqueeze(1)
        normals = normals_norm.reshape((bs, nf, 3))

        if direction.dim() == 2:
            direction = direction.unsqueeze(1)
        cos = torch.nn.functional.relu(torch.sum(normals * direction, dim=2))
        light = (light + intensity_directional * color_directional.unsqueeze(1)
                 * cos.unsqueeze(2))

    # apply
    light = light.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    textures = textures * light
    return textures
