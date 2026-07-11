"""
Shared utilities. The one that matters most: load_or_compute.

In the original notebooks, every stage looked like:
    drive.mount(...)
    df = pd.read_pickle(some_path)     # if it existed
    ... do expensive work ...
    df.to_pickle(some_path)            # save it again
repeated with slightly different paths in four separate files, which made it
easy to lose track of which file was the current version of what.

load_or_compute centralizes that pattern in one place: give it a path and a
function that produces the result, and it handles the "do I already have this
cached?" check consistently everywhere it's used.
"""

import os
import pickle


def load_or_compute(path, compute_fn, force_recompute=False):
    """Load a pickled object from `path` if it exists, otherwise compute it
    with `compute_fn()` and cache it to `path` for next time.

    Args:
        path: where the cached result lives (or will be saved).
        compute_fn: a zero-argument callable that produces the result.
        force_recompute: if True, ignore any existing cache and recompute.
    """
    if not force_recompute and os.path.exists(path):
        print(f"[cache hit] loading {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"[cache miss] computing result for {path}")
    result = compute_fn()

    with open(path, "wb") as f:
        pickle.dump(result, f)
    print(f"[cached] saved to {path}")

    return result
