# -*- coding: utf-8 -*-
"""CT intensity normalization and optional resampling helpers.

Functions provided:
- `normalize_CT(image: np.ndarray) -> np.ndarray` : min-max normalize to [0,1] (float32)
- `lumTrans(image, lungwin=(-1200., 600.)) -> np.ndarray` : apply HU window and map to [0,255] uint8
- `resample_sitk_image_to_spacing(sitk_image, target_spacing)` : resample a SimpleITK Image to target spacing
- `sitk_image_to_numpy(sitk_image)` : convert SimpleITK Image to numpy array and return (array, origin, spacing, direction)

These provide a single place to control intensity normalization and optional spatial resampling.
"""
from typing import Tuple, Optional
import numpy as np
import SimpleITK as sitk


def normalize_CT(image: np.ndarray) -> np.ndarray:
    """Min-max normalize image to range [0,1] and return float32 array.

    Operates in float32 to save memory on large volumes.
    If image is constant (max == min) returns float32 copy.
    """
    img = np.asarray(image, dtype=np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx == mn:
        return img
    out = (img - mn) / (mx - mn)
    return out.astype(np.float32)


def lumTrans(image: np.ndarray, lungwin: Tuple[float, float] = (-1200., 600.)) -> np.ndarray:
    """Apply lung window (HU) and map to 0-255 uint8 similar to original implementation.

    Parameters
    - image: input image in HU (numpy)
    - lungwin: (min, max) window in HU
    """
    img = np.asarray(image, dtype=np.float32)
    low, high = float(lungwin[0]), float(lungwin[1])
    newimg = (img - low) / (high - low)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255.0).astype(np.uint8)
    return newimg


def resample_sitk_image_to_spacing(sitk_image: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
    """Resample a SimpleITK Image to the given target spacing (x,y,z).

    Returns a new SimpleITK Image with the requested spacing using linear interpolation.
    """
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    # compute new size rounding to nearest int
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / float(target_spacing[i]))))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(sitk_image)


def sitk_image_to_numpy(sitk_image: sitk.Image) -> Tuple[np.ndarray, Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Convert SimpleITK Image to numpy array and return (array, origin, spacing, direction).

    The array is returned in shape (D,H,W) consistent with existing code.
    Origin/spacing/direction are returned in numpy-friendly order (reversed where appropriate as used elsewhere).
    """
    arr = sitk.GetArrayFromImage(sitk_image)
    origin = list(reversed(sitk_image.GetOrigin()))
    spacing = list(reversed(sitk_image.GetSpacing()))
    direction = list(reversed(sitk_image.GetDirection()))
    return arr, origin, spacing, direction
