# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import datetime
import argparse

import numpy as np
import h5py
import cv2

import turret
import turret.caffe
import turret.layers as L


MAX_BATCH_SIZE = 128
WORKSPACE_SIZE = 2**30
IMAGE_SIZE = 224


def preprocess(image, size, mean):
    in_h, in_w, _ = image.shape
    crop_size = min(in_h, in_w)
    crop_y = (in_h - crop_size) // 2
    crop_x = (in_w - crop_size) // 2
    image = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32)
    image = (image.transpose(2, 0, 1) - mean)
    offset = (256 - size) // 2;
    image = image[:, offset:offset+size, offset:offset+size]
    return image.astype(np.float32)


def parse_image_list(listfile):
    basedir = os.path.split(listfile)[0]
    with open(listfile, "r") as f:
        return [os.path.join(basedir, p.strip()) for p in f]

def create_int8_calibrator(listfile, img_size, mean, count=None):
    def sample_generator(flist):
        for fname in flist:
            image = preprocess(cv2.imread(fname), IMAGE_SIZE, mean)
            yield image
    flist = parse_image_list(listfile)
    if count is not None:
        flist = flist[:count]
    return turret.Int8Calibrator(sample_generator(flist), MAX_BATCH_SIZE)

def run_build(args):
    if args.dtype == "int8" and args.calibrator is None:
        sys.stderr.write("calibrator is required for int8 inference.\n")
        sys.exit(-1)
    if args.dtype == "int8":
        dtype = turret.DataType.INT8
        calibrator = create_int8_calibrator(
            args.calibrator, IMAGE_SIZE, np.load(args.mean), args.nbatches)
    elif args.dtype == "half":
        dtype = turret.DataType.HALF
        calibrator = None
    elif args.dtype == "float":
        dtype = turret.DataType.FLOAT
        calibrator = None

    logger = turret.loggers.ConsoleLogger()
    builder = turret.InferenceEngineBuilder(logger)
    network = builder.create_network(dtype)
    tensor_set = turret.caffe.import_caffemodel(
        network, args.deploy, args.model)
    network.mark_output("prob", tensor_set["prob"])
    builder.max_batch_size = MAX_BATCH_SIZE
    builder.max_workspace_size = WORKSPACE_SIZE
    builder.int8_calibrator = calibrator
    engine = builder.build(network)
    with open(args.dest, "wb") as f:
        engine.serialize(f)


def load_synset_words(listfile):
    with open(listfile, "r") as f:
        return [line.split(None, 1)[1].strip() for line in f]

def run_predict(args):
    words = load_synset_words(args.words)
    mean = np.load(args.mean)
    logger = turret.loggers.ConsoleLogger()
    runtime = turret.InferenceRuntime(logger)
    with open(args.model, "rb") as f:
        engine = runtime.deserialize_engine(f)
    with turret.ExecutionContext(engine) as ctx:
        buf = ctx.create_buffer()
        for fname in args.inputs:
            print("{}:".format(fname))
            image = preprocess(cv2.imread(fname), IMAGE_SIZE, mean)
            buf.put("data", image.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE))
            ctx.execute(buf)
            prob = buf.get("prob").reshape(len(words))
            for k in prob.argsort()[-5:][::-1]:
                print("{} ({:.1f}%)".format(words[k], prob[k] * 100.0))


def run_benchmark(args):
    mean = np.load(args.mean)
    logger = turret.loggers.ConsoleLogger()
    runtime = turret.InferenceRuntime(logger)
    with open(args.model, "rb") as f:
        engine = runtime.deserialize_engine(f)
    with turret.ExecutionContext(engine) as ctx:
        buf = ctx.create_buffer()
        img_size = buf.get("data").shape[2]
        image = preprocess(cv2.imread(args.input), img_size, mean)
        buf.put("data", np.broadcast_to(image, (args.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)))
        # warm-up
        ctx.execute(buf)
        # benchmark
        start_dt = datetime.datetime.now()
        for i in range(args.loop):
            ctx.execute(buf)
        end_dt = datetime.datetime.now()
        duration = (end_dt - start_dt).total_seconds()
        print("{} frames in {} seconds: {} FPS".format(
            args.loop, duration, args.loop / duration))


def main():
    parser = argparse.ArgumentParser(description="Turret importing caffemodel example")
    subcommands = parser.add_subparsers()

    build_parser = subcommands.add_parser("build")
    build_parser.add_argument("-d", "--dtype", default="float",
                              choices=["float", "half", "int8"],
                              help="data type for computations in inference")
    build_parser.add_argument("-c", "--calibrator",
                              help="image set for calibration")
    build_parser.add_argument("-n", "--nbatches", type=int, default=32,
                              help="number of batches for calibration")
    build_parser.add_argument("--mean", default="mean.npy", help="mean image")
    build_parser.add_argument("deploy", help="path to deploy.prototxt")
    build_parser.add_argument("model", help="path to caffemodel")
    build_parser.add_argument("dest", help="destination to serialize model")
    build_parser.set_defaults(handler=run_build)

    predict_parser = subcommands.add_parser("predict")
    predict_parser.add_argument("-m", "--model", required=True,
                                help="serialized model")
    predict_parser.add_argument("--mean", default="mean.npy", help="mean image")
    predict_parser.add_argument("-w", "--words", default="synset_words.txt",
                                help="synset word list")
    predict_parser.add_argument("inputs", nargs="+")
    predict_parser.set_defaults(handler=run_predict)

    benchmark_parser = subcommands.add_parser("benchmark")
    benchmark_parser.add_argument("-m", "--model", required=True,
                                  help="serialized model")
    benchmark_parser.add_argument("--mean", default="mean.npy", help="mean image")
    benchmark_parser.add_argument("-l", "--loop", type=int, default=200)
    benchmark_parser.add_argument("-b", "--batch_size", type=int, default=64)
    benchmark_parser.add_argument("input")
    benchmark_parser.set_defaults(handler=run_benchmark)

    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
