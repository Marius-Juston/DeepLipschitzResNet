"""
Entry point for Lipschitz-aware model components.

Parameters
----------
None

Returns
-------
None

Raises
------
ImportError
    Raised if downstream modules fail to load due to missing dependencies.

Notes
-----
Exposes reusable building blocks such as :mod:`models.model` (end-to-end
assemblers), :mod:`models.cnn_layers` (spectrally normalized convolutions), and
:mod:`models.linear_layers` (Lipschitz linear blocks).

Mathematical Sketch
-------------------
Aggregating modules ensures that composed networks maintain bounded Lipschitz
constants because each exported component enforces its own spectral constraint.

Examples
--------
>>> from models.model import LipschitzNetwork
>>> LipschitzNetwork
<class 'models.model.LipschitzNetwork'>
"""
