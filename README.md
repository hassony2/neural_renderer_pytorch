Neural Renderer Pytorch
=======================

Pytorch implementation of [Neural Renderer](https://github.com/hiroharu-kato/neural_renderer)

Mostly this is a reimplementation of most utility functions provided py Neural Renderer.
Most functions are an almost line-to-line translation of the original code and each function
([rasterize](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/rasterize.py), [perspective](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/perspective.py), ...)
have a pytorch equivalent ([rasterize_th](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/rasterize_th.py) [perspective_th](https://github.com/hassony2/neural_renderer_pytorch/blob/master/neurender/perspective_th.py))

Some cleaning is still needed for a better separation of pytorch and chainer code, so that the pytorch code can be used without a chainer install).

Compared to the original directory, the structure of the code is changes, and respects the folder structure.

# Demo

`python examples/example_2_th.py`

If you find this code useful for your research, please cite the original work 

```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
