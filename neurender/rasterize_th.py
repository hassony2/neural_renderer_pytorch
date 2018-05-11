import numpy as np
import torch
from torch.autograd import gradcheck, Function


class Rasterize(Function):
    def __init__(self,
                 image_size,
                 near,
                 far,
                 eps,
                 background_color,
                 return_rgb=False,
                 return_alpha=False,
                 return_depth=False):
        super().__init__()

        if not any((return_rgb, return_alpha, return_depth)):
            # nothing to draw
            raise Exception

        # arguments
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

        # input buffers
        self.faces = None
        self.textures = None
        self.grad_rgb_map = None
        self.grad_alpha_map = None
        self.grad_depth_map = None

        # output buffers
        self.rgb_map = None
        self.alpha_map = None
        self.depth_map = None
        self.grad_faces = None
        self.grad_textures = None

        # intermediate buffers
        self.face_index_map = None
        self.weight_map = None
        self.face_inv_map = None
        self.sampling_index_map = None
        self.sampling_weight_map = None

        # input information
        self.xp = None
        self.batch_size = None
        self.num_faces = None
        self.texture_size = None

    @staticmethod
    def forward(ctx, inputs):
        import pdb
        pdb.set_trace()
        return inputs

    @staticmethod
    def backward(ctx, grad_out):
        return None


def rasterize_rgbad(faces,
                    textures=None,
                    image_size=256,
                    anti_aliasing=False,
                    near=0.1,
                    far=100,
                    eps=1e-3,
                    background_color=[0, 0, 0],
                    return_rgb=True,
                    return_alpha=True,
                    return_depth=True):
    if textures is None:
        inputs = [faces]
    else:
        raise ValueError('textures not implemented')

    if anti_aliasing:
        raise ValueError('anti_aliasing not implemented')
    else:
        rasterize = Rasterize(image_size, near, far, eps, background_color,
                              return_rgb, return_alpha, return_depth)
        import pdb
        pdb.set_trace()
        rgb, alpha, depth = rasterize.apply(*inputs)
        ret = {
            'rgb': rgb if return_rgb else None,
            'alpha': alpha if return_alpha else None,
            'depth': depth if return_depth else None
        }
        return ret


def rasterize_silhouettes(faces,
                          image_size=256,
                          anti_aliasing=False,
                          near=0.1,
                          far=100,
                          eps=1e-3):
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far,
                           eps, None, False, True, False)['alpha']


def get_line_coeff(pt_1, pt_2):
    if pt_1[0] == pt_2[0]:
        a = 1
        b = 0
        c = -pt_1[0].item()
        return a, b, c
    if pt_1[1] == pt_2[1]:
        a = 0
        b = 1
        c = -pt_1[1].item()
        return a, b, c
    a = pt_1[1] - pt_2[1]
    b = pt_2[0] - pt_1[0]
    c = pt_2[1] * pt_1[0] - pt_1[1] * pt_2[0]
    return a.item(), b.item(), c.item()


def get_line_coeffs(face):
    a1, b1, c1 = get_line_coeff(face[1], face[0])
    a2, b2, c2 = get_line_coeff(face[2], face[1])
    a3, b3, c3 = get_line_coeff(face[0], face[2])
    return a1, b1, c1, a2, b2, c2, a3, b3, c3


def rasterize_face_forward(face,
                           image_height=10,
                           image_width=10,
                           background=0,
                           face_color=1):
    img = torch.zeros(image_height, image_width)
    min_y, min_x = face.min(0)[0]
    max_y, max_x = face.max(0)[0]
    a1, b1, c1, a2, b2, c2, a3, b3, c3 = get_line_coeffs(face)
    for y in range(int(min_y.item()), int(max_y.item())):
        for x in range(int(min_x.item()), int(max_x.item())):
            if a1 * y + b1 * x + c1 >= 0:
                if a2 * y + b2 * x + c2 >= 0:
                    if a3 * y + b3 * x + c3 >= 0:
                        img[y, x] = 1
    return img


def order_points(pts):
    """
    Return in order idxs of leftmost middle and rightmos points
    """
    assert pts.shape[0] == 3
