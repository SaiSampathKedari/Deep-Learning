"""Project paths, resolved once.

Defining these in a single module — rather than scattering
``Path(__file__).parents[N]`` through the codebase — means moving a file can
only ever break one line instead of many.

``DATA_ROOT`` follows the convention used by torch (``TORCH_HOME``), HuggingFace
(``HF_HOME``) and Keras (``KERAS_HOME``): an environment variable overrides a
sensible default, so the data location can be changed without editing code.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DL_DATA_ROOT", REPO_ROOT / "data" / "raw"))
