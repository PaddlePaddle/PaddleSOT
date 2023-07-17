import abc
import codecs
import collections
import contextlib
import copy
import copyreg
import dataclasses
import distutils
import enum
import functools
import importlib
import inspect
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import sre_compile
import sre_parse
import sys
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import uuid
import weakref

import _collections_abc
import _weakrefset
import decorator
import google.protobuf
import numpy
import setuptools

from ..utils import log


def _strip_init_py(s):
    return re.sub(r"__init__.py$", "", s)


def _module_dir(m: types.ModuleType):
    return _strip_init_py(m.__file__)


skip_file_names = {
    _module_dir(m)
    for m in (
        abc,
        collections,
        contextlib,
        copy,
        copyreg,
        dataclasses,
        enum,
        functools,
        google.protobuf,
        importlib,
        inspect,
        linecache,
        logging,
        multiprocessing,
        numpy,
        operator,
        os,
        posixpath,
        random,
        re,
        selectors,
        sre_compile,
        sre_parse,
        signal,
        tempfile,
        threading,
        tokenize,
        traceback,
        types,
        typing,
        unittest,
        weakref,
        _collections_abc,
        _weakrefset,
        decorator,
        codecs,
        uuid,
        setuptools,
        distutils,
    )
}


sot_path = os.path.dirname(__file__).rpartition("/")[0] + "/"
paddle_path = sys.modules["paddle"].__file__.rpartition("/")[0] + "/"

skip_file_names.add(sot_path)
skip_file_names.add(paddle_path)
skip_file_names.add(
    "<frozen importlib",
)
skip_file_names.add("<__array_function__ internals>")

skip_file_name_re = re.compile(
    f"^({'|'.join(map(re.escape, skip_file_names))})"
)

no_skip_file_names = {paddle_path + 'nn/layer/container.py'}


customed_skip_code = set()


def need_skip_path(filepath: str) -> bool:
    """
    Check if the file should be skipped and not transcribed.

    Args:
        filepath: The path of the file to check.

    Returns:
        bool: True if the file should be skipped.
    """
    if filepath in no_skip_file_names:
        return False
    if not filepath.startswith("<"):
        filepath = os.path.abspath(filepath)
    return bool(skip_file_name_re.match(filepath))


def skip_function(function):
    customed_skip_code.add(function.__code__)


def need_skip(pycode):
    if pycode in customed_skip_code:
        log(3, f"Skip frame by code: {pycode}")
        return True
    return need_skip_path(pycode.co_filename)
