"""Compatibility shims applied once, on import of :mod:`efgp`.

jax-finufft <= 1.3.1 batching rule compares batch axes against
``jax._src.interpreters.batching.not_mapped``. Recent JAX releases (>= 0.10)
removed that name and use ``None`` for an unmapped axis instead. Without the
alias, ``jax.vmap`` over a NUFFT (for example over trials, or when lineax
materialises an operator) fails with an ``AttributeError``.

Remove this module once jax-finufft ships a release with the fix.
"""

from jax._src.interpreters import batching as _batching

if not hasattr(_batching, "not_mapped"):
    _batching.not_mapped = None
