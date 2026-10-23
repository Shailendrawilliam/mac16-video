import mlx.core as mx


def _f(name):
    return getattr(mx, name, None) or getattr(mx.metal, name)


def reset_peak():
    _f("reset_peak_memory")()


def peak_gb():
    return _f("get_peak_memory")() / 1e9
