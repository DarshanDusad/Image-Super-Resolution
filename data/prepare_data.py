# Responsible for preparing the dataset
import argparse
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if img.size[0] != size:
        # Just good practice, not required
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    return [lr_img, hr_img, sr_img]


def resize_worker(img_file, sizes, resample):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return img_file.name.split('.')[0], out


# Context shared by each worker
class WorkingContext():
    def __init__(self, resize_fn, out_path, sizes):
        self.resize_fn = resize_fn
        self.out_path = out_path
        self.sizes = sizes

        # Shared counter
        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value


def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img = imgs
        lr_img.save('{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
        hr_img.save('{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)))
        sr_img.save('{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1], i.zfill(5)))
        curr_total = wctx.inc_get()


# Checks if all threads are finished with their job
def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True


def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample=Image.BICUBIC):
    # Resampling refers to the process of changing the resolution or size of a
    # digital image by adding or removing pixels.
    # Bicubic interpolation is one of the methods used to resample images.
    # The algorithm estimates the values of the new pixels
    # based on the surrounding pixels using a mathematical function.

    # Partial functions allow us to fix a certain number of arguments of a function and generate a new function
    # Resize function now only takes one argument
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)
    files = [p for p in Path('{}'.format(img_path)).glob(f'**/*')]

    # Make three directories - Low Resolution images, High Resolution Images, Super Resolution Images
    os.makedirs(out_path, exist_ok=True)
    os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
    os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
    os.makedirs('{}/sr_{}_{}'.format(out_path, sizes[0], sizes[1]), exist_ok=True)

    if n_worker > 1:
        # Split files for each worker
        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, out_path, sizes)

        #  Start worker process
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)

        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='{}/datasets/celeba_hq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str, default='./train/celeba_hq_16_128')
    parser.add_argument('--size', type=str, default='16,128')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker, sizes=sizes, resample=resample)
