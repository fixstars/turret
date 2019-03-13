# -*- coding: utf-8 -*-

registered_classes = {}

def auto_register(cls):
    """The function for decorator. Users need not to use."""
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)
    registered_classes[cls.module_name()] = cls
    return wrapper
