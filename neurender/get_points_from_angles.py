import math

import chainer
import numpy as np
import torch


def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
                -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        xp = chainer.cuda.get_array_module(distance)
        if degrees:
            elevation = xp.radians(elevation)
            azimuth = xp.radians(azimuth)
        return xp.stack([
            distance * xp.cos(elevation) * xp.sin(azimuth),
            distance * xp.sin(elevation),
            -distance * xp.cos(elevation) * xp.cos(azimuth),
        ]).transpose()


def get_points_from_angles_th(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
                -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = np.radians(elevation)
            azimuth = np.radians(azimuth)
        return np.stack([
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
            -distance * np.cos(elevation) * np.cos(azimuth),
        ]).transpose()
