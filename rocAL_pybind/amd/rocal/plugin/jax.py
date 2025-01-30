# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

##
# @file jax.py
#
# @brief File containing iterators and functions for JAX framework

import jax
import numpy as np
from packaging.version import Version

import rocal_pybind as b
import amd.rocal.types as types

if Version(jax.__version__) < Version("0.4.23"):
    print('rocAL only supports jax versions >= 0.4.23')


def convert_to_jax_array(array):
    """Converts input array to JAX array.

    Args:
        rocal_tensor (rocalTensor):
            array to be converted to JAX array.

    Returns:
        jax.Array: JAX array with the same values and device as input array.
    """
    jax_array = jax.dlpack.from_dlpack(array)
    return jax_array

class ROCALJaxIterator(object):
    """!Iterator for processing data

        @param pipeline            The rocAL pipeline to use for processing data.
        @param device              The device to use for processing
        @param device_id           The ID of the device to use
    """

    def __init__(self, pipeline, device="cpu", device_id=0):
        self.loader = pipeline
        self.device = device
        self.device_id = device_id
        self.batch_size = pipeline._batch_size
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        self.labels_size = ((self.batch_size * self.loader._num_classes)
                            if self.loader._one_hot_encoding else self.batch_size)
        self.dimensions = self.dtype = None
        self.labels_tensor = None
        self.iterator_length = b.getRemainingImages(
            self.loader._handle) // self.batch_size  # iteration length
        if self.loader._is_external_source_operator:
            self.eos = False
            self.index = 0
            self.num_batches = self.loader._external_source.n // self.batch_size if self.loader._external_source.n % self.batch_size == 0 else (
                self.loader._external_source.n // self.batch_size + 1)

    def next(self):
        return self.__next__()

    def __next__(self):
        if (self.loader._is_external_source_operator):
            if (self.index + 1) == self.num_batches:
                self.eos = True
            if (self.index + 1) <= self.num_batches:
                data_loader_source = next(self.loader._external_source)
                # Extract all data from the source
                images_list = data_loader_source[0] if (self.loader._external_source_mode == types.EXTSOURCE_FNAME) else []
                input_buffer = data_loader_source[0] if (self.loader._external_source_mode != types.EXTSOURCE_FNAME) else []
                labels_data = data_loader_source[1] if (len(data_loader_source) > 1) else None
                roi_height = data_loader_source[2] if (len(data_loader_source) > 2) else []
                roi_width = data_loader_source[3] if (len(data_loader_source) > 3) else []
                ROIxywh_list = []
                for i in range(self.batch_size):
                    ROIxywh = b.ROIxywh()
                    ROIxywh.x =  0
                    ROIxywh.y =  0
                    ROIxywh.w = roi_width[i] if len(roi_width) > 0 else 0
                    ROIxywh.h = roi_height[i] if len(roi_height) > 0 else 0
                    ROIxywh_list.append(ROIxywh)
                if (len(data_loader_source) == 6 and self.loader._external_source_mode == types.EXTSOURCE_RAW_UNCOMPRESSED):
                    decoded_height = data_loader_source[4]
                    decoded_width = data_loader_source[5]
                else:
                    decoded_height = self.loader._external_source_user_given_height
                    decoded_width = self.loader._external_source_user_given_width

                kwargs_pybind = {
                    "handle": self.loader._handle,
                    "source_input_images": images_list,
                    "labels": labels_data,
                    "input_batch_buffer": input_buffer,
                    "roi_xywh": ROIxywh_list,
                    "decoded_width": decoded_width,
                    "decoded_height": decoded_height,
                    "channels": 3,
                    "external_source_mode": self.loader._external_source_mode,
                    "rocal_tensor_layout": types.NCHW,
                    "eos": self.eos}
                b.externalSourceFeedInput(*(kwargs_pybind.values()))
            self.index = self.index + 1
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()

        self.output_list = []
        for i in range(len(self.output_tensor_list)):
            self.dimensions = self.output_tensor_list[i].dimensions()
            self.dtype = self.output_tensor_list[i].dtype()
            self.output = convert_to_jax_array(self.output_tensor_list[i])
            self.output_list.append(self.output)
        if (self.loader._is_external_source_operator):
            self.labels = self.loader.get_image_labels()
            self.labels_tensor = self.labels.astype(dtype=np.int_)
            return self.output_list, self.labels_tensor

        if self.loader._name == "labelReader":
            if self.loader._one_hot_encoding == True:
                self.labels = np.empty(self.labels_size, dtype="int32")
                self.loader.get_one_hot_encoded_labels(
                        self.labels.ctypes.data, self.loader._output_memory_type)
                self.labels_tensor = self.labels.reshape(
                    -1, self.batch_size, self.loader._num_classes)
                self.labels_tensor = convert_to_jax_array(self.labels_tensor)
            else:
                self.labels = self.loader.get_image_labels()
                self.labels_tensor = self.labels.astype(dtype=np.int_)
                self.labels_tensor = convert_to_jax_array(self.labels_tensor)

        return self.output_list, self.labels_tensor

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.iterator_length

    def __del__(self):
        b.rocalRelease(self.loader._handle)