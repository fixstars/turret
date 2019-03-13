# -*- coding: utf-8 -*-
import unittest
import turret

import numpy as np


class Int8Test(unittest.TestCase):
    def _create_builder(self):
        builder = turret.InferenceEngineBuilder(
            turret.loggers.ConsoleLogger())
        return builder

    def _create_network(self, builder):
        network = builder.create_network(turret.DataType.INT8)
        h = network.add_input("input", turret.DataType.FLOAT,
                              turret.Dimensions.CHW(1, 1, 1))
        h = turret.layers.relu(h)
        network.mark_output("output", h)
        return network

    def test_successful_calibration(self):
        SAMPLE_COUNT = 64
        def generator():
            for i in range(SAMPLE_COUNT):
                generator.fetched += 1
                yield np.asarray([[[i]]], dtype=np.float32)
        generator.fetched = 0
        builder = self._create_builder()
        builder.max_workspace_size = 1 << 28
        builder.int8_calibrator = \
            turret.Int8Calibrator(generator(), 1)
        builder.build(self._create_network(builder))
        self.assertEqual(SAMPLE_COUNT, generator.fetched)

    def test_shape_mismatch(self):
        builder = self._create_builder()
        builder.max_workspace_size = 1 << 28
        builder.int8_calibrator = turret.Int8Calibrator(
            [np.asarray([[[1, 2]]], dtype=np.float32)], 1)
        self.assertRaises(
            ValueError,
            lambda: builder.build(self._create_network(builder)))

    def test_type_mismatch(self):
        builder = self._create_builder()
        builder.max_workspace_size = 1 << 28
        builder.int8_calibrator = turret.Int8Calibrator(
            [np.asarray([[[1]]], dtype=np.float64)], 1)
        self.assertRaises(
            TypeError,
            lambda: builder.build(self._create_network(builder)))

    def test_error_propagation(self):
        def generator():
            yield np.asarray([[[1]]], dtype=np.float32)
            raise RuntimeError()
        builder = self._create_builder()
        builder.max_workspace_size = 1 << 28
        builder.int8_calibrator = \
            turret.Int8Calibrator(generator(), 1)
        self.assertRaises(
            RuntimeError,
            lambda: builder.build(self._create_network(builder)))
