import contextlib
import keyword
from typing import Callable, Dict, List, Set, Optional, Union, Iterable

from fastapi.encoders import jsonable_encoder
import httpx
from loguru import logger

from leopardai._internal.client_utils import (  # noqa
    _get_method_docstring,
    _get_positional_argument_error_message,
)
from leopardai.api.connection import Connection
from leopardai.api.v1.workspace_record import WorkspaceRecord
from leopardai.api.util import (
    _get_full_workspace_url,
    _get_full_workspace_api_url,
)
from leopardai.config import DEFAULT_PORT
from leopardai.photon import FileParam  # noqa
from leopardai.util import is_valid_url
from .api import deployment, APIError


def local(port: int = DEFAULT_PORT) -> str:
    """
    Create a connection string for a local deployment. This is useful for testing
    purposes, and does not require you to type the local IP address repeatedly.

    Usage:
        client = Client(local())
        client.foo()

    Args:
        port (int, optional): The port number. Defaults to leopardai.config.DEFAULT_PORT.
    Returns:
        str: The connection string.
    """
    return f"http://localhost:{port}"


def current() -> str:
    """
    Returns the current workspace id. This is useful for creating a client that
    calls deployments in the current workspace. Note that when instantiating a
    client, if the current workspace is used, the token will be automatically
    set to the current workspace token if not specified.
    """
    id = WorkspaceRecord.get_current_workspace_id()
    if id is None:
        raise RuntimeError("No current workspace is set.")
    return id


class _MultipleEndpointWithDefault(object):
    """
    A class that wraps multiple endpoints with the same name and different http
    methods, with a default. DO NOT USE THIS CLASS DIRECTLY. This is an internal
    utility class.
    """

    def __init__(self, default: Callable, http_method: str):
        """
        A class that represents a multiple endpoint with a default method. This is
        used when the client encounters multiple endpoints with the same name.
        """
        self._default = default
        self._default_http_method = http_method
        self.methods: Dict[str, Callable] = {http_method: default}

    def default_http_method(self) -> str:
        return self._default_http_method

    def _add_method(
        self, method: Callable, http_method: str
    ) -> "_MultipleEndpointWithDefault":
        """
        Adds a method to the multiple endpoint. This is used when the client
        encounters multiple endpoints with the same name. Returns self for
        chaining purposes.
        """
        self.methods[http_method] = method
        return self

    def __call__(self, *args, **kwargs):
        return self._default(*args, **kwargs)

    def __getattr__(self, http_method: str) -> Callable:
        if http_method not in self.methods:
            raise AttributeError(f"No http method called {http_method} exists.")
        return self.methods[http_method]


class PathTree(object):
    def __init__(self, name: str, debug_record: List):
        self._path_cache: Dict[str, Union[Callable, PathTree]] = {}
        self._method_cache: Dict[str, str] = {}
        self._all_paths: Set[str] = set()
        self.name = name
        self.debug_record = debug_record

    def __getattr__(self, name: str) -> Union[Callable, "PathTree"]:
        try:
            return self._path_cache[name]
        except KeyError:
            raise AttributeError(
                f"No such path named {name} found. I am currently at {self.name} and"
                f" available members are ({','.join(self._path_cache.keys())})."
            )

    def __len__(self):
        return len(self._path_cache)

    def __dir__(self) -> Iterable[str]:
        return self._path_cache.keys()

    def __getitem__(self, name: str) -> Union[Callable, "PathTree"]:
        try:
            return self._path_cache[name]
        except KeyError:
            raise AttributeError(
                f"No such path named {name} found. I am currently at {self.name} and"
                f" available members are ({','.join(self._path_cache.keys())})."
            )

    def __setitem__(
        self, name: str, value: Union[Callable, "PathTree"]
    ) -> None:  # noqa
        raise NotImplementedError(
            "PathTree does not support dictionary-type set. Use add(path, func)"
            " instead."
        )

    def __call__(self):
        paths_ordered = list(self._all_paths)
        paths_ordered.sort()
        path_separator = "\n- "
        return (
            "A wrapper for leopardai Client that contains the following paths:\n"
            f"- {path_separator.join(paths_ordered)}\n"
        )

    def _has(self, path_or_name: str) -> bool:
        return path_or_name in self._all_paths or path_or_name in self._path_cache

    @staticmethod
    def rectify_name(name: str) -> str:
        """
        Rectifies the path to be a valid python identifier. For example,
        "foo/bar" will be converted to "foo_bar".
        """
        if keyword.iskeyword(name):
            name += "_"
        return name.replace("-", "_").replace(".", "_")

    # implementation note: prefixing this function with "_" in case there is
    # an api function that is called "add". Ditto for "_has" above.
    def _add(self, path: str, func: Callable, http_method: str) -> None:
        """
        Adds a path to the path tree. The path can contain "/"s, in which each "/"
        will be split and converted to children nodes.
        """
        # Record all paths for bookkeeping purposes.
        self._all_paths.add(path)
        # Remove the leading and trailing slashes, which are not needed.
        path = path.strip("/")

        if path == "":
            # This is the default path that the client will call with "__call__()".
            if "" in self._path_cache:
                # already existing path: convert the current path to a multiple
                # endpoint with default.
                self._path_cache[""] = _MultipleEndpointWithDefault(
                    self._path_cache[""], self._method_cache[""]
                )._add_method(func, http_method)
                self._method_cache[""] = "__multiple__"
            else:
                self._path_cache[""] = func
                self._method_cache[""] = http_method
        elif "/" in path:
            prefix, remaining = path.split("/", 1)
            prefix = self.rectify_name(prefix)
            if prefix.isidentifier():
                if prefix not in self._path_cache:
                    self._path_cache[prefix] = PathTree(
                        (self.name + "." if self.name else "") + prefix,
                        self.debug_record,
                    )
                    self._method_cache[prefix] = "__intermediate__"
                self._path_cache[prefix]._add(remaining, func, http_method)
            else:
                # temporarily ignore this path if it is not a valid identifier.
                # this is to prevent the case where the path is something like
                # "foo/{bar}" which we don't support yet.
                self.debug_record.append(
                    f"Prefix {path} is not a valid identifier. Ignoring for now."
                )
                return
        else:
            path = self.rectify_name(path)
            if path.isidentifier():
                if path in self._path_cache:
                    new_method = _MultipleEndpointWithDefault(
                        self._path_cache[path], self._method_cache[path]
                    )._add_method(func, http_method)
                    self._path_cache[path] = new_method
                    self._method_cache[path] = "__multiple__"
                else:
                    self._path_cache[path] = func
                    self._method_cache[path] = http_method
            else:
                # temporarily ignore this path if it is not a valid identifier.
                # this is to prevent the case where the path is something like
                # "foo/{bar}" which we don't support yet.
                self.debug_record.append(
                    f"Path {path} is not a valid identifier. Ignoring for now."
                )
                return
