"""Tests never trace: switch Langfuse off before any app module reads the setting."""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
