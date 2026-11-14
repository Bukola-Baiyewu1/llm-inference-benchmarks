"""When does sharding a matmul need communication? Checked in the compiled program.

y = x @ w is split across a 4 x 2 mesh of (simulated) devices in two ways:

* ``batch``: rows of x are split. Each device computes its own rows of y.
  No communication is needed.
* ``contraction``: the summed-over dimension is split (columns of x, rows of w).
  Each device holds only a partial sum, so XLA must add them across devices:
  an all-reduce appears in the compiled program.

Runs on CPU by simulating 8 devices, so no TPU is needed.

python -m infbench.jax_sharding --out results/jax_sharding.json
"""
from __future__ import annotations

import os

# Must be set before JAX starts: pretend this CPU is 8 devices.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import argparse  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

COMM_OPS = ("all-reduce", "reduce-scatter", "all-gather", "collective-permute", "all-to-all")


def make_mesh() -> Mesh:
    # Always use the 8 simulated CPU devices, even on a machine with real GPUs
    # (e.g. Kaggle's 2 x T4), so the 4 x 2 mesh exists everywhere.
    cpus = jax.devices("cpu")
    if len(cpus) < 8:
        raise RuntimeError(f"need 8 CPU devices, found {len(cpus)}: XLA_FLAGS must be set before JAX starts")
    devices = np.array(cpus[:8]).reshape(4, 2)
    return Mesh(devices, axis_names=("data", "model"))


def sharded_matmul(mode: str, m: int = 256, k: int = 512, n: int = 128) -> dict:
    mesh = make_mesh()
    # Keep every array on the CPU devices, so nothing mixes with a real GPU if one exists.
    with jax.default_device(jax.devices("cpu")[0]):
        kx, kw = jax.random.split(jax.random.PRNGKey(0))
        x = jax.random.normal(kx, (m, k)); w = jax.random.normal(kw, (k, n))
        reference = x @ w
    if mode == "batch":
        x_spec, w_spec = P("data", None), P(None, None)
    elif mode == "contraction":
        x_spec, w_spec = P(None, ("data", "model")), P(("data", "model"), None)
    else:
        raise ValueError(mode)
    xs = jax.device_put(x, NamedSharding(mesh, x_spec))
    ws = jax.device_put(w, NamedSharding(mesh, w_spec))
    f = jax.jit(lambda a, b: a @ b)
    hlo = f.lower(xs, ws).compile().as_text()
    y = f(xs, ws)
    found = sorted({op for op in COMM_OPS if op in hlo})
    return {"mode": mode, "x_sharding": str(x_spec), "w_sharding": str(w_spec),
            "communication_ops": found, "has_all_reduce": "all-reduce" in hlo,
            "matches_unsharded": bool(jnp.allclose(jax.device_get(y), jax.device_get(reference), atol=1e-3)),
            "output_sharding": str(y.sharding.spec)}


def main() -> None:
    from .common import env_info, save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/jax_sharding.json")
    a = ap.parse_args()
    print(f"JAX {jax.__version__}: using {len(jax.devices('cpu'))} simulated CPU devices for a 4 x 2 mesh")
    rows = [sharded_matmul("batch"), sharded_matmul("contraction")]
    for r in rows:
        print(f"{r['mode']:12s} comm ops: {r['communication_ops'] or 'none'}  matches unsharded: {r['matches_unsharded']}")
    print("saved", save_json({"env": env_info(), "jax": jax.__version__, "rows": rows}, a.out))


if __name__ == "__main__":
    main()
