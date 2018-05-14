"""
Example 2. Optimizing vertices.
"""
import argparse
import glob
import os
import subprocess

import chainer
import chainer.functions as cf
import numpy as np
import scipy.misc
import torch
import tqdm

from neurender.get_points_from_angles import get_points_from_angles_th
from neurender.renderer_th import Renderer
from neurender.load_obj import load_obj


class Model(torch.nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super().__init__()

        vertices, faces = load_obj(filename_obj)
        self.vertices = vertices
        self.faces = faces

        # create textures
        texture_size = 2
        textures = np.ones((1, self.faces.shape[1], texture_size, texture_size,
                            texture_size, 3), 'float32')
        self.textures = textures

        # load reference image
        self.image_ref = torch.Tensor(
            scipy.misc.imread(filename_ref).astype('float32').mean(-1) / 255.)

        # setup renderer
        renderer = Renderer()
        self.renderer = renderer

    def forward(self):
        self.renderer.eye = get_points_from_angles_th(2.732, 0, 90)
        image = self.renderer.render_silhouettes(self.vertices, self.faces)
        loss = cf.sum(cf.square(image - self.image_ref[None, :, :]))
        return loss


def make_gif(working_directory, filename):
    # generate gif (need ImageMagick)
    options = '-delay 8 -loop 0 -layers optimize'
    subprocess.call(
        'convert %s %s/_tmp_*.png %s' % (options, working_directory, filename),
        shell=True)
    for filename in glob.glob('%s/_tmp_*.png' % working_directory):
        os.remove(filename)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-io', '--filename_obj', type=str, default='examples/data/teapot.obj')
    parser.add_argument(
        '-ir',
        '--filename_ref',
        type=str,
        default='./examples/data/example2_ref.png')
    parser.add_argument(
        '-oo',
        '--filename_output_optimization',
        type=str,
        default='./examples/data/example2_optimization.gif')
    parser.add_argument(
        '-or',
        '--filename_output_result',
        type=str,
        default='./examples/data/example2_result.gif')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    working_directory = os.path.dirname(args.filename_output_result)

    vertices, faces = load_obj(args.filename_obj)
    vertices = torch.Tensor(vertices).cuda().unsqueeze(0)
    vertices.requires_grad = True
    faces = torch.Tensor(faces).cuda().unsqueeze(0)
    faces.requires_grad = True

    # create textures
    texture_size = 2
    textures = faces.new_full(
        (1, faces.shape[1], texture_size, texture_size, texture_size, 3),
        fill_value=1)

    # load reference image
    image_ref = torch.Tensor(
        scipy.misc.imread(args.filename_ref).astype('float32').mean(-1) / 255.)
    image_ref = image_ref.cuda()

    # setup renderer
    renderer = Renderer()

    optimizer = torch.optim.Adam([vertices, faces])
    loop = tqdm.tqdm(range(300))
    for i in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        renderer.eye = get_points_from_angles_th(2.732, 0, 90)
        images = renderer.render_silhouettes(vertices, faces)
        loss = torch.sum((images - image_ref)**2)
        loss.backward()
        optimizer.step()
        image = images.detach().cpu().numpy()[0]
        scipy.misc.toimage(
            image, cmin=0,
            cmax=1).save('%s/_tmp_%04d.png' % (working_directory, i))
    make_gif(working_directory, args.filename_output_optimization)

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = get_points_from_angles_th(2.732, 0, azimuth)
        images = renderer.render(vertices, faces, textures)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        scipy.misc.toimage(
            image, cmin=0,
            cmax=1).save('%s/_tmp_%04d.png' % (working_directory, num))
    make_gif(working_directory, args.filename_output_result)


if __name__ == '__main__':
    run()
