Neural Renderer Pytorch
=======================

Pytorch implementation of [Neural Renderer](https://github.com/hiroharu-kato/neural_renderer)

Another (better optimized !) implementation that does not require chainer is available by [Nikos Kolotouros](https://github.com/daniilidis-group/neural_renderer) 

Mostly this code is a reimplementation of most utility functions provided py Neural Renderer.
Most functions are an almost line-to-line translation of the original code and each function
([rasterize](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/rasterize.py), [perspective](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/perspective.py), ...)
have a pytorch equivalent ([rasterize_th](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/rasterize_th.py) [perspective_th](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/perspective_th.py))

Some cleaning is still needed for a better separation of pytorch and chainer code.

Compared to the original directory, the structure of the code is changed, and respects the folder structure.

The tests have also been reproduced for the pytorch functions.

For the forward/backward pass, a wrapper around chainer is used, therefore [chainer](https://docs.chainer.org/en/stable/install.html) and ([PyTorch](https://pytorch.org/) of course as well) need to be installed ! 

# Demo

`python examples/example_2_th.py`
