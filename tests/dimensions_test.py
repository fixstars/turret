# -*- coding: utf-8 -*-
import unittest
import turret

class DimensionsTest(unittest.TestCase):
    def test_init_form_dimension(self):
        dims = turret.Dimensions((
            turret.Dimension(11, turret.DimensionType.CHANNEL),
            turret.Dimension(17, turret.DimensionType.SPATIAL)))
        self.assertEqual(2, len(dims))
        self.assertEqual(11, dims[0].size)
        self.assertEqual(17, dims[1].size)
        self.assertEqual(turret.DimensionType.CHANNEL, dims[0].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[1].type)

    def test_init_from_tuple(self):
        dims = turret.Dimensions((
            (13, turret.DimensionType.INDEX),
            ( 7, turret.DimensionType.CHANNEL),
            (19, turret.DimensionType.SPATIAL)))
        self.assertEqual(3, len(dims))
        self.assertEqual(13, dims[0].size)
        self.assertEqual( 7, dims[1].size)
        self.assertEqual(19, dims[2].size)
        self.assertEqual(turret.DimensionType.INDEX,   dims[0].type)
        self.assertEqual(turret.DimensionType.CHANNEL, dims[1].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[2].type)

    def test_init_from_invalid_types(self):
        def from_valid_and_int():
            return turret.Dimensions(((2, turret.DimensionType.SPATIAL), 3))
        self.assertRaises(TypeError, from_valid_and_int)

    def test_init_from_invalid_tuples(self):
        def from_too_large_tuple():
            return turret.Dimensions((
                (3, turret.DimensionType.SPATIAL, None)))
        self.assertRaises(TypeError, from_too_large_tuple)
        def from_too_small_tuple():
            return turret.Dimensions(((1,), (2,)))
        self.assertRaises(TypeError, from_too_small_tuple)

    def test_iterator(self):
        dims = turret.Dimensions((
            (13, turret.DimensionType.SEQUENCE),
            ( 7, turret.DimensionType.CHANNEL),
            (19, turret.DimensionType.SPATIAL)))
        it = iter(dims)
        dim = next(it)
        self.assertEqual(13, dim.size)
        self.assertEqual(turret.DimensionType.SEQUENCE, dim.type)
        dim = next(it)
        self.assertEqual( 7, dim.size)
        self.assertEqual(turret.DimensionType.CHANNEL, dim.type)
        dim = next(it)
        self.assertEqual(19, dim.size)
        self.assertEqual(turret.DimensionType.SPATIAL, dim.type)
        self.assertRaises(StopIteration, lambda: next(it))

    def test_shape(self):
        dims = turret.Dimensions((
            (13, turret.DimensionType.SEQUENCE),
            ( 7, turret.DimensionType.CHANNEL),
            (19, turret.DimensionType.SPATIAL)))
        self.assertEqual((13, 7, 19), dims.shape)
        dims = turret.Dimensions((
            (19, turret.DimensionType.SPATIAL),))
        self.assertEqual((19,), dims.shape)

    def test_size(self):
        dims = turret.Dimensions((
            (13, turret.DimensionType.SEQUENCE),
            ( 7, turret.DimensionType.CHANNEL),
            (19, turret.DimensionType.SPATIAL)))
        self.assertEqual(13*7*19, dims.size)

    def test_hw(self):
        dims = turret.Dimensions.HW(101, 103)
        self.assertEqual(2, len(dims))
        self.assertEqual(101, dims[0].size)
        self.assertEqual(103, dims[1].size)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[0].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[1].type)

    def test_chw(self):
        dims = turret.Dimensions.CHW(107, 101, 103)
        self.assertEqual(3, len(dims))
        self.assertEqual(107, dims[0].size)
        self.assertEqual(101, dims[1].size)
        self.assertEqual(103, dims[2].size)
        self.assertEqual(turret.DimensionType.CHANNEL, dims[0].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[1].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[2].type)

    def test_nchw(self):
        dims = turret.Dimensions.NCHW(113, 107, 101, 103)
        self.assertEqual(4, len(dims))
        self.assertEqual(113, dims[0].size)
        self.assertEqual(107, dims[1].size)
        self.assertEqual(101, dims[2].size)
        self.assertEqual(103, dims[3].size)
        self.assertEqual(turret.DimensionType.INDEX,   dims[0].type)
        self.assertEqual(turret.DimensionType.CHANNEL, dims[1].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[2].type)
        self.assertEqual(turret.DimensionType.SPATIAL, dims[3].type)
