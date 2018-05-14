import chainer
import cupy as cp
import numpy as np
import torch
from torch.autograd import gradcheck, Function, Variable

from neurender.rasterize import Rasterize
from neurender.rasterize import rasterize_silhouettes as rasterize_silhouettes_ch
from neurender.rasterize import rasterize as rasterize_rgb_ch


class RasterizeSil(torch.nn.Module):
    def __init__(self, image_size, near, far, eps, background_color):
        super().__init__()
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.rasterize = lambda faces: rasterize_silhouettes_ch(faces, image_size, False, near, far, eps)

    def forward(self, inputs):
        out = RasterizeSilF.apply(inputs, self.rasterize)
        return out


class RasterizeSilF(Function):
    @staticmethod
    def forward(ctx, faces, rasterize):
        ctx.rasterize = rasterize
        faces = chainer.Variable(faces.detach().cpu().numpy())
        faces.to_gpu()
        ctx.inputs = (faces, )
        outputs = rasterize(faces)
        ctx.outputs = outputs
        outputs = Variable(torch.Tensor(outputs.data.get()).cuda())
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        grad_cp = cp.asarray(grad_out.cpu().numpy())
        chainer.cuda.to_gpu(grad_cp)
        ctx.outputs.grad = grad_cp
        ctx.outputs.backward()
        back = ctx.inputs[0].grad.get()
        back = torch.Tensor(back)
        return back, None


class RasterizeRGB(torch.nn.Module):
    def __init__(self, image_size, near, far, eps, background_color):
        super().__init__()
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.rasterize = lambda faces, textures: rasterize_rgb_ch(faces, textures, image_size, False, near, far, eps, background_color)

    def forward(self, faces, textures):
        out = RasterizeRGBF.apply(faces, textures, self.rasterize)
        return out


class RasterizeRGBF(Function):
    @staticmethod
    def forward(ctx, faces, textures, rasterize):
        ctx.rasterize = rasterize
        faces = chainer.Variable(faces.detach().cpu().numpy())
        textures = chainer.Variable(textures.detach().cpu().numpy())
        faces.to_gpu()
        textures.to_gpu()
        ctx.inputs = (faces, textures)
        outputs = rasterize(faces, textures)
        ctx.outputs = outputs
        outputs = Variable(torch.Tensor(outputs.data.get()).cuda())
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        grad_cp = cp.asarray(grad_out.cpu().numpy())
        chainer.cuda.to_gpu(grad_cp)
        ctx.outputs.grad = grad_cp
        ctx.outputs.backward()

        # Get vertices gradients
        back_verts = ctx.inputs[0].grad.get()
        back_verts = torch.Tensor(back_verts)

        # Get textures gradients
        back_text = ctx.inputs[1].grad.get()
        back_text = torch.Tensor(back_text)
        return back_verts, back_text, None
