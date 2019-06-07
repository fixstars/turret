# -*- coding: utf-8 -*-
import os
import sys
import numpy

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand
from distutils.sysconfig import customize_compiler
from Cython.Build import cythonize
from Cython.Distutils import build_ext

sys.path.append("./tests")

class cpp_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

setup(name="turret",
      version="5.0.0",
      cmdclass={
          "build_ext": cpp_build_ext,
          "test": PyTest
      },
      packages=[
          "turret",
          "turret.nvinfer",
          "turret.loggers",
          "turret.layers",
          "turret.plugin",
          "turret.int8",
          "turret.caffe"
      ],
      ext_modules=cythonize([
          "turret/logger.pyx",
          "turret/logger_bridge.pyx",
          "turret/foundational.pyx",
          "turret/graph.pyx",
          "turret/buffer.pyx",
          "turret/engine.pyx",
          "turret/int8/cy_calibrator_proxy.pyx",
          "turret/int8/calibrator_proxy_bridge.pyx",
          "turret/layers/builtin.pyx",
          "turret/plugin/cy_plugin_proxy.pyx",
          "turret/plugin/plugin_proxy_bridge.pyx",
          "turret/plugin/cy_plugin_factory_proxy.pyx",
          "turret/plugin/plugin_factory_proxy_bridge.pyx",
          "turret/plugin/temporary_stream_context.pyx",
          "turret/caffe/caffe.pyx",
          "turret/caffe/cy_plugin_factory_proxy.pyx",
          "turret/caffe/plugin_factory_proxy_bridge.pyx",
      ]),
      include_dirs=[numpy.get_include()],
      install_requires=[
          "numpy",
          "pycuda",
          "six"
      ],
      tests_require=["pytest"],
      test_suite="tests")
