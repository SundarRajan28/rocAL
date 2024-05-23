from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.pytorch import ROCALNumpyIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import os, glob


MEAN = [0.026144592091441154, -88.3379898071289, -84.62094116210938, -78.56366729736328, -77.72217559814453, 7.33015557974337e-12, 48330.79296875, 87595.4296875, 183.57638549804688, 208.38265991210938, -7.185957863625792e-19, 109.64270782470703, 94.19403076171875, -0.37584438920021057, 9952.041015625, 20.362579345703125]
STDDEV = [108.9710922241211, 174.1948699951172, 173.99221801757812, 155.323486328125, 158.25418090820312, 0.14563894271850586, 58919.42578125, 24443.921875, 64.71000671386719, 77.63092041015625, 3.7348792830016464e-05, 242.97598266601562, 237.60250854492188, 5726.51611328125, 2953.1953125, 51.31494903564453]

def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data

def get_data_split(path: str):
    imgs = load_data(path, "data-*.npy")
    lbls = load_data(path, "label-*.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    return imgs, lbls

def main():
    if  len(sys.argv) < 3:
        print ('Please pass numpy_folder cpu/gpu batch_size')
        exit(0)
    try:
        path= "OUTPUT_IMAGES_PYTHON/NEW_API/NUMPY_READER/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
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
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_train, shard_id=local_rank, num_shards=world_size)
        label_output = fn.readers.numpy(file_root=data_path, files=y_train, shard_id=local_rank, num_shards=world_size)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NHWC)
        normalized_output = fn.normalize(data_output, axes=[0,1], mean=MEAN, stddev=STDDEV, output_datatype=types.FLOAT)
        transposed_output = fn.transpose(normalized_output, perm=[2,0,1], output_layout=types.NCHW, output_dtype=types.FLOAT)
        pipeline.set_outputs(transposed_output, label_output)

    pipeline.build()

    val_pipeline = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, prefetch_queue_depth=6)

    with val_pipeline:
        numpy_reader_output = fn.readers.numpy(file_root=data_path, files=x_val, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        label_output = fn.readers.numpy(file_root=data_path, files=y_val, shard_id=local_rank, num_shards=world_size, seed=random_seed+local_rank)
        data_output = fn.set_layout(numpy_reader_output, output_layout=types.NHWC)
        normalized_output = fn.normalize(data_output, axes=[0,1], mean=MEAN, stddev=STDDEV, output_datatype=types.FLOAT)
        transposed_output = fn.transpose(normalized_output, perm=[2,0,1], output_layout=types.NCHW, output_dtype=types.FLOAT)
        val_pipeline.set_outputs(transposed_output, label_output)

    val_pipeline.build()
    
    numpyIteratorPipeline = ROCALNumpyIterator(pipeline, device='cpu' if rocal_cpu else 'gpu')
    print(len(numpyIteratorPipeline))
    valNumpyIteratorPipeline = ROCALNumpyIterator(val_pipeline, device='cpu' if rocal_cpu else 'gpu')
    print(len(valNumpyIteratorPipeline))
    cnt = 0
    for epoch in range(2):
        print("+++++++++++++++++++++++++++++EPOCH+++++++++++++++++++++++++++++++++++++",epoch)
        for i , it in enumerate(numpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        numpyIteratorPipeline.reset()
        for i , it in enumerate(valNumpyIteratorPipeline):
            print(i, it[0].shape, it[1].shape)
            for j in range(batch_size):
                print(it[0][j].cpu().numpy().shape, it[1][j].cpu().numpy().shape)
                cnt += 1
            print("************************************** i *************************************",i)
        valNumpyIteratorPipeline.reset()
    print("*********************************************************************")
    print(f'Took {time.time() - start} seconds')

if __name__ == '__main__':
    main()
