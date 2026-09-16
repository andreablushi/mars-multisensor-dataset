"""Keeping the DigitalHub SDK from crashing on a credential that holds a %.

Python loads this module as it starts, since a job puts the scripts root on its
path, so the patch is in place before the platform authenticates the job. The
SDK parses its credentials file with `%` interpolation, which rejects such a
credential, so the file is read and written raw instead. Delete this module once
the SDK stops interpolating it.
"""

from __future__ import annotations

from configparser import RawConfigParser

try:
    from digitalhub.stores.client.auth import file_module
except ImportError:
    file_module = None

if file_module is not None:
    file_module.ConfigParser = RawConfigParser
