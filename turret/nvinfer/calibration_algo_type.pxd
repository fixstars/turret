from .nvinfer cimport CalibrationAlgoType

cdef extern from "NvInfer.h" namespace "nvinfer1::CalibrationAlgoType":
    cdef CalibrationAlgoType kLEGACY_CALIBRATION
    cdef CalibrationAlgoType kENTROPY_CALIBRATION
