"""Model implementations — one module per family.

Each model exposes a stable interface so a shared training loop can drive any of
them: ``params`` (dict of tensors) and ``loss(X, y=None)`` returning
``(loss, grads)`` when ``y`` is given, or scores when it is not.
"""
