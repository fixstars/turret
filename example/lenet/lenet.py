# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import h5py

import turret
import turret.layers as L

import mnist


BATCH_SIZE = 128
CALIBRATOR_COUNT = 1024


def main():
    images = mnist.get_test_images()
    labels = mnist.get_test_labels()
    n_samples = images.shape[0]
    images = images.reshape((n_samples, 1) + images.shape[1:])

    logger = turret.loggers.ConsoleLogger()
    builder = turret.InferenceEngineBuilder(logger)
    network = builder.create_network(turret.DataType.INT8)
    with h5py.File("model.h5", "r") as h5file:
        def param(name):
            return h5file["predictor"][name][:]
        h = network.add_input("input", turret.DataType.FLOAT,
                              turret.Dimensions.CHW(1, 28, 28))
        h = L.convolution_2d(h, param("conv1/W"), param("conv1/b"))
        h = L.max_pooling_2d(h, 2, stride=2)
        h = L.convolution_2d(h, param("conv2/W"), param("conv2/b"))
        h = L.max_pooling_2d(h, 2, stride=2)
        h = L.fully_connected(h, param("fc1/W"), param("fc1/b"))
        h = L.relu(h)
        h = L.fully_connected(h, param("fc2/W"), param("fc2/b"))
        h = L.softmax(h)
        network.mark_output("prob", h)
        builder.max_batch_size = BATCH_SIZE
        builder.max_workspace_size = 2 ** 30
        builder.int8_calibrator = turret.Int8Calibrator(
            images[:CALIBRATOR_COUNT], BATCH_SIZE)
        engine = builder.build(network)

    ctx = turret.ExecutionContext(engine)
    buf = ctx.create_buffer()
    n_correct = 0
    for head in range(0, n_samples, BATCH_SIZE):
        tail = min(head + BATCH_SIZE, n_samples)
        buf.put("input", images[head:tail])
        ctx.execute(buf)
        prob = buf.get("prob").reshape((tail-head, 10))
        prediction = prob.argmax(axis=1)
        n_correct += np.equal(prediction, labels[head:tail]).sum()
    print("Accuracy: {}".format(n_correct / n_samples))


if __name__ == "__main__":
    main()
