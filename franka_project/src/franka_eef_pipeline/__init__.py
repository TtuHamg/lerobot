"""Project-local Franka EEF utilities.

The package root stays intentionally lightweight. Import the concrete module
that owns a capability, for example ``franka_eef_pipeline.geometry`` or
``franka_eef_pipeline.async_server``. This keeps inference-only deployments
from importing MCAP/data-conversion dependencies they do not use.
"""
