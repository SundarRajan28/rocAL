# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import os, sys
import random
from functools import partial
import cv2
import numpy as np


def crop_fn(img, crop_size):
    return img[:, :crop_size[0], :crop_size[1], :]    # Crop along the height and width dimensions

def flip_fn(img):
    rand_prob = random.random()
    if rand_prob < 0.5:
        return img[:, :, ::-1, :]
    else:
        return img

def blend_images(image1, image2):
    # Overlays image2 onto image1 in a circular mask for NHWC arrays.
    assert image1.shape == image2.shape
    n, h, w, c = image1.shape
    y, x = np.ogrid[0:h, 0:w]                               # Create the coordinate grids
    mask = (x - w / 2) ** 2 + (y - h / 2) ** 2 > h * w / 9  # Create the circular mask
    result1 = np.copy(image1)
    result1[:, mask, :] = image2[:, mask, :]
    return result1

def draw_patches(image, idx, layout="nchw", dtype="fp32", device="cpu"):
    # image is expected as a numpy array
    if layout == "nchw":
        image = image.transpose([1, 2, 0])
    if dtype in ["fp16", "fp32"]:
        image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_folder/python_function/" + str(idx) +
                "_" + "train" + ".png", image)


def main():
    if len(sys.argv) < 3:
        print("Please pass image_folder batch_size")
        exit(0)
    try:
        path = "output_folder/python_function/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    data_path = sys.argv[1]
    rocal_cpu = True  # Only supported for Host backend
    batch_size = int(sys.argv[2])
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    local_rank = 0
    world_size = 1

    crop_image_fn = partial(crop_fn, crop_size=(224, 224))
    # Pipeline example with crop + hflip + blend augmentations
    pipe = Pipeline(batch_size=batch_size, num_threads=8, device_id=local_rank,
                                                   seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC, tensor_dtype=types.UINT8)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_path)
        decode = fn.decoders.image(jpegs, file_root=data_path, output_type=types.RGB, shard_id=local_rank, num_shards=world_size, random_shuffle=False)
        cropped_output = fn.python_function(decode, function = crop_image_fn, output_dims=(224, 224, 3), dtype=types.UINT8, layout=types.NHWC)
        flipped_output = fn.python_function(cropped_output, function = flip_fn, output_dims=(224, 224, 3), dtype=types.UINT8, layout=types.NHWC)
        blend_output = fn.python_function(cropped_output, flipped_output, function = blend_images, output_dims=(224, 224, 3), dtype=types.UINT8, layout=types.NHWC)
        pipe.set_outputs(blend_output)
    pipe.build()
    
    # Dataloader
    data_loader = ROCALClassificationIterator(pipe, device="cpu", device_id=local_rank)
    cnt = 0

    # Enumerate over the Dataloader
    for epoch in range(3):
        print(
            "+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++", epoch)
        for i, it in enumerate(data_loader):
            print(
                "************************************** i *************************************", i)
            for img in it[0]:
                cnt += 1
                draw_patches(img[0], cnt, layout="nhwc",
                             dtype="fp32", device=rocal_cpu)
        data_loader.reset()
    print("##############################  PYTHON FUNCTION OPERATOR SUCCESS  ############################")



if __name__ == "__main__":
    main()
