"""Server package for OpenEnv deployment.

Re-exports `app` so runners that use `uvicorn server:app` work.
"""

from .app import app  # noqa: F401

