# flake8: noqa
"""
This implements the CLI for the leopard AI library. When you install the library,
you get a command line tool called `lep` that you can use to operate local photon
including creating, managing, and running them, and to interact with the cloud.
"""

# Guard so that leopardai.api never depends on things under leopardai.cli.
import leopardai.api as _

from .cli import lep