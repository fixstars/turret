# -*- coding: utf-8 -*-
import os
import struct
import gzip
import requests
import numpy as np

class DownloadError(Exception):
    pass

def _download(dest, src):
    r = requests.get(src)
    if r.status_code != 200:
        raise DownloadError()
    with open(dest, 'wb') as f:
        f.write(r.content)

def get_test_images():
    FILENAME = 't10k-images-idx3-ubyte.gz'
    URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    if not os.path.exists(FILENAME):
        _download(FILENAME, URL)
    with gzip.open(FILENAME, 'rb') as f:
        magic, n, h, w = struct.unpack('>IIII', f.read(16))
        if magic != 0x0803:
            raise ValueError()
        fmt = 'B' * (h * w)
        images = []
        for i in range(n):
            pixels = struct.unpack(fmt, f.read(h * w))
            images.append(list(map(lambda x: x / 255.0, pixels)))
        return np.array(images, dtype=np.float32).reshape((n, h, w))

def get_test_labels():
    FILENAME = 't10k-labels-idx1-ubyte.gz'
    URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    if not os.path.exists(FILENAME):
        _download(FILENAME, URL)
    with gzip.open(FILENAME, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        if magic != 0x0801:
            raise ValueError()
        return np.array(struct.unpack('B' * n, f.read(n)), dtype=np.int32)
