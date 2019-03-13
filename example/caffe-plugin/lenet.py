# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import h5py

import turret
import turret.layers as L

import mnist
import fully_connected


BATCH_SIZE = 128


def fully_connected_factory(name, weights):
    # weights[0]: weigts
    # weights[1]: bias
    return fully_connected.FullyConnectedPlugin(weights[0], weights[1])

def main():
    images = mnist.get_test_images()
    labels = mnist.get_test_labels()
    n_samples = images.shape[0]
    images = images.reshape((n_samples, 1) + images.shape[1:])

    # scale from [0.0, 1.0] to [255.0, 0.0]
    images = (1.0 - images) * 255.0
    # subtract mean values
    # TODO load mean image from binraryproto
    images = images - np.mean(images, axis=0)

    logger = turret.loggers.ConsoleLogger()
    builder = turret.InferenceEngineBuilder(logger)
    network = builder.create_network(turret.DataType.FLOAT)
    plugin_factory = turret.caffe.PluginFactory()
    plugin_factory.register_plugin("ip2", fully_connected_factory)
    tensors = turret.caffe.import_caffemodel(
            network, "mnist.prototxt", "mnist.caffemodel", plugin_factory)
    network.mark_output("prob", tensors["prob"])
    builder.max_batch_size = BATCH_SIZE
    engine = builder.build(network)

    ctx = turret.ExecutionContext(engine)
    buf = ctx.create_buffer()
    n_correct = 0
    for head in range(0, n_samples, BATCH_SIZE):
        tail = min(head + BATCH_SIZE, n_samples)
        buf.put("data", images[head:tail])
        ctx.execute(buf)
        prob = buf.get("prob").reshape((tail-head, 10))
        prediction = prob.argmax(axis=1)
        n_correct += np.equal(prediction, labels[head:tail]).sum()
    print("Accuracy: {}".format(n_correct / n_samples))


if __name__ == "__main__":
    main()
