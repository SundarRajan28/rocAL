from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os
import numpy as np

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
import amd.rocal.fn as fn
import sys
import os, glob


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def get_data_split(path: str):
    imgs = load_data(path, "*data*.npy")
    lbls = load_data(path, "*label*.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    return imgs, lbls

def main():
    if  len(sys.argv) < 4:
        print ('Please pass numpy_folder numpy_folder1 cpu/gpu batch_size')
        exit(0)
    data_path = sys.argv[1]
    data_path1 = sys.argv[2]
    if(sys.argv[3] == "cpu"):
        rocal_cpu = True
    else:
        rocal_cpu = False
    batch_size = int(sys.argv[4])
    num_threads = 8
    device_id = 0
    local_rank = 0
    world_size = 1
    random_seed = random.SystemRandom().randint(0, 2**32 - 1)
    x_train, y_train = get_data_split(data_path)
    x_val, y_val = get_data_split(data_path1)

    import time
    start = time.time()
    pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=6)

    with pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_train, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        label_output = fn.readers.numpy(file_root=data_path, files=y_train, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NDHWC)
        log_add_output = fn.log(data_output)
        pipeline.set_outputs(log_add_output, label_output)

    pipeline.build()

    val_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=6)

    with val_pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path1, files=x_val, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        label_output = fn.readers.numpy(file_root=data_path1, files=y_val, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NDHWC)
        log_add_output = fn.log(data_output)
        val_pipeline.set_outputs(log_add_output, label_output)

    val_pipeline.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, device='cpu' if rocal_cpu else 'gpu', device_id=device_id)
    print(len(numpyIteratorPipeline))
    valNumpyIteratorPipeline = ROCALNumpyIterator(val_pipeline, device='cpu' if rocal_cpu else 'gpu', device_id=device_id)
    print(len(valNumpyIteratorPipeline))

    for epoch in range(1):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(numpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape, np.allclose(it[0][0].cpu(), np.log1p(np.load(x_train[i])).astype(np.float32)))
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
        for i , it in enumerate(valNumpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape, np.allclose(it[0][0].cpu(), np.log1p(np.load(x_val[i])).astype(np.float32)))
            print("************************************** i *************************************",i)
        valNumpyIteratorPipeline.reset()
    print("*********************************************************************")


if __name__ == '__main__':
    main()
