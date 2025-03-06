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

from matplotlib import category
import jax
import jax.dlpack
import jax.numpy as jnp
from jax.sharding import NamedSharding, PositionalSharding, Sharding

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

    def __init__(self, pipelines, sharding=None):
        if not isinstance(pipelines, list):
            pipelines = [pipelines]
        self.pipelines = pipelines
        self.num_devices = len(pipelines)
        self.batch_size = pipelines[0]._batch_size

        self.dimensions = self.dtype = None
        self.labels_tensor = None
        self.iterator_length = b.getRemainingImages(
            pipelines[0]._handle) // self.batch_size  # Length should be the same across all pipelines

        if sharding is not None:
            assert isinstance(
                sharding, (NamedSharding, PositionalSharding)
            ), "`sharding` should be an instance of `NamedSharding` or `PositionalSharding`"
        self._sharding = sharding

    def next(self):
        return self.__next__()

    def __next__(self):
        pipeline_outputs = []
        for self.loader in self.pipelines:
            if self.loader.rocal_run() != 0:
                raise StopIteration
            self.output_tensor_list = self.loader.get_output_tensors()

            self.output_list = []
            self.device_id = self.loader._device_id
            if self.loader._name is None:
                self.loader._name = self.loader._reader
            self.last_batch_policy = self.loader._last_batch_policy
            assert (
                self.last_batch_policy != types.LAST_BATCH_PARTIAL
            ), "JAX iterator does not support partial last batch policy."
            self.labels_size = ((self.batch_size * self.loader._num_classes)
                                if self.loader._one_hot_encoding else self.batch_size)
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.dtype = self.output_tensor_list[i].dtype()
                self.output = convert_to_jax_array(
                    self.output_tensor_list[i].__dlpack__(self.device_id))
                self.output_list.append(self.output)

            if self.loader._name == "labelReader":
                if self.loader._one_hot_encoding == True:
                    self.labels = np.empty(self.labels_size, dtype="int32")
                    self.loader.get_one_hot_encoded_labels(
                        self.labels.ctypes.data, self.loader._output_memory_type)
                    self.labels_tensor = self.labels.reshape(
                        -1, self.batch_size, self.loader._num_classes)
                    self.labels_tensor = convert_to_jax_array(
                        self.labels_tensor)
                else:
                    self.labels = self.loader.get_image_labels()
                    self.labels_tensor = self.labels.astype(dtype=np.int_)
                    self.labels_tensor = convert_to_jax_array(
                        self.labels_tensor)
                self.labels_tensor = jax.device_put(
                    self.labels_tensor, self.output_list[0].device)
                self.output_list.append(self.labels_tensor)
            pipeline_outputs.append(self.output_list)

        if self.num_devices == 1 and self._sharding is None:
            return pipeline_outputs[0]

        sharded_outputs = []
        for i in range(len(pipeline_outputs[0])):
            individual_outputs = []
            for pipeline_id in range(self.num_devices):
                individual_outputs.append(pipeline_outputs[pipeline_id][i])
            for output in individual_outputs:
                assert output.shape == individual_outputs[0].shape, "All outputs should have the same shape"
            if self._sharding is not None:
                sharded_outputs.append(self.place_output_with_sharding(
                    individual_outputs))
            else:
                sharded_outputs.append(self.place_output_with_device_put(
                    individual_outputs))
        return sharded_outputs

    def reset(self):
        for self.loader in self.pipelines:
            b.rocalResetLoaders(self.loader._handle)

    def place_output_with_device_put(self, individual_outputs):
        """Builds sharded jax.Array with `jax.device_put_sharded` - compatible
        with pmapped JAX functions.
        """
        output_devices = tuple(
            map(lambda jax_shard: jax_shard.device, individual_outputs)
        )

        if len(output_devices) != len(set(output_devices)):
            if len(set(output_devices)) != 1:
                raise AssertionError(
                    "JAX iterator requires shards to be placed on \
                                                different devices or all on the same device."
                )
            else:
                # All shards are on one device (CPU or one GPU)
                return jnp.stack(individual_outputs)
        else:
            return jax.device_put_sharded(individual_outputs, output_devices)

    def place_output_with_sharding(self, individual_outputs):
        """Builds sharded jax.Array with `jax.make_array_from_single_device_arrays`-
        compatible with automatic parallelization with JAX.
        """
        shard_shape = individual_outputs[0].shape

        if isinstance(self._sharding, NamedSharding):
            global_shape = (self._sharding.mesh.size *
                            shard_shape[0], *shard_shape[1:])
        else:
            global_shape = (
                self._sharding.shape[0] * shard_shape[0], *shard_shape[1:])

        return jax.make_array_from_single_device_arrays(
            global_shape, self._sharding, individual_outputs
        )

    def __iter__(self):
        return self

    def __len__(self):
        return self.iterator_length

    def __del__(self):
        for self.loader in self.pipelines:
            b.rocalRelease(self.loader._handle)
