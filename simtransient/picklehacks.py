"""
Add a pickle function for serializing methods which are bound to a class.

This allows use of multiprocessing (indirectly, via emcee's parallel threading
options) when the probability functions are members of a class structure.

More complex pickle functionality is available via e.g. dill/pathos, but this
works fine for now and keeps the dependencies minimal.
"""

import copy_reg
import types

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)