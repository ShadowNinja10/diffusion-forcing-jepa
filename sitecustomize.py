import numpy as np

# Compatibility shim for packages (e.g., imgaug) that still rely on np.sctypes.
# NumPy 2 removed np.sctypes from the public namespace.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "float": (np.float16, np.float32, np.float64, np.longdouble),
        "int":   (np.int8, np.int16, np.int32, np.int64, np.longlong),
        "uint":  (np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong),
    }
