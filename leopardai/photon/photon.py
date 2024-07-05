from abc import abstractmethod
from collections import OrderedDict
import copy
import functools
import cloudpickle
import importlib
import importlib.util
import inspect
import logging
import os
import re
import signal
import sys
import threading
import time
import traceback
from types import FunctionType, FrameType
from typing import Callable, Any, List, Dict, Optional, Iterator, Type
from typing_extensions import Annotated
import warnings
import zipfile

import anyio
import click
from fastapi import APIRouter, FastAPI, HTTPException, Body, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import (  # noqa: F401
    Response,
    JSONResponse,
    FileResponse,
    StreamingResponse,
)
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
import pydantic
import pydantic.decorator
import uvicorn
import uvicorn.config

from leopardai.config import (  # noqa: F401
    ALLOW_ORIGINS,
    BASE_IMAGE,
    BASE_IMAGE_ARGS,
    BASE_IMAGE_CMD,
    DEFAULT_PORT,
    DEFAULT_INCOMING_TRAFFIC_GRACE_PERIOD,
    DEFAULT_TIMEOUT_KEEP_ALIVE,
    DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN,
    ENV_VAR_REQUIRED,
    PYDANTIC_MAJOR_VERSION,
    VALID_SHAPES,
    get_local_deployment_token,
)
from leopardai.photon.constants import METADATA_VCS_URL_KEY
from leopardai.photon.download import fetch_code_from_vcs
from leopardai.photon.types import (  # noqa: F401
    File,
    FileParam,
    PNGResponse,
    JPEGResponse,
    WAVResponse,
)
from leopardai.util import switch_cwd, patch, asyncfy_with_semaphore
from leopardai.util.cancel_on_disconnect import run_with_cancel_on_disconnect
from .base import BasePhoton, schema_registry
from .batcher import batch
import leopardai._internal.logging as internal_logging

schemas = ["py"]


def create_model_for_func(
    func: Callable, func_name: str, use_raw_args: bool, http_method: str
):
    (
        args,
        _,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    ) = inspect.getfullargspec(func)
    if len(args) > 0 and (args[0] == "self" or args[0] == "cls"):
        args = args[1:]  # remove self or cls

    if not use_raw_args and (args or varkw):
        if defaults is None:
            defaults = ()
        non_default_args_count = len(args) - len(defaults)
        defaults = (...,) * non_default_args_count + defaults

        keyword_only_params = {
            param: kwonlydefaults.get(param, Any) for param in kwonlyargs
        }
        params = {
            param: (annotations.get(param, Any), default)
            for param, default in zip(args, defaults)
        }

        if varkw:
            if PYDANTIC_MAJOR_VERSION <= 1:

                class config:  # type: ignore
                    extra = "allow"

            else:
                config = pydantic.ConfigDict(extra="allow")  # type: ignore
        else:
            config = None  # type: ignore

        func_name = func_name or func.__name__
        request_model = pydantic.create_model(
            f"{func_name.capitalize()}{http_method.capitalize()}Input",
            **params,
            **keyword_only_params,
            __config__=config,  # type: ignore
        )
    else:
        request_model = None

    return_type = inspect.signature(func).return_annotation

    if inspect.isclass(return_type) and issubclass(return_type, Response):
        response_model = None
        response_class = return_type
    else:
        if return_type is inspect.Signature.empty:
            return_type = Any

        if PYDANTIC_MAJOR_VERSION <= 1:

            class config:
                arbitrary_types_allowed = True

        else:
            config = pydantic.ConfigDict(arbitrary_types_allowed=True)  # type: ignore

        response_model = pydantic.create_model(
            f"{func_name.capitalize()}{http_method.capitalize()}Output",
            output=(return_type, None),
            __config__=config,  # type: ignore
        )
        response_class = JSONResponse
    return request_model, response_model, response_class


PHOTON_HANDLER_PARAMS_KEY = "__photon_handler_params__"


# A utility lock for the photons to use when running the _call_init_once function.
# The reason it is not inside the Photon class is that, the Photon class is going to be
# cloudpickled, and cloudpickle does not work with threading.Lock.
# A downside is that, if the user creates multiple photons in the same process, and they
# all call _call_init_once at the same time, they will be serialized by this lock. This is
# not a big deal, because the init function is usually not a bottleneck.
_photon_initialize_lock = threading.Lock()


class Photon(BasePhoton):
    photon_type: str = "photon"
    obj_pkl_filename: str = "obj.pkl"
    py_src_filename: str = "py.py"

    # Required python dependencies that you usually install with `pip install`. For example, if
    # your photon depends on `numpy`, you can set `requirement_dependency=["numpy"]`. If your
    # photon depends on a package installable over github, you can set the dependency to
    # `requirement_dependency=["git+xxxx"] where `xxxx` is the url to the github repo.
    #
    # Experimental feature: if you specify "uninstall xxxxx", instead of installing the library,
    # we will uninstall the library. This is useful if you want to uninstall a library that is
    # in conflict with some other libraries, and need to sequentialize a bunch of pip installs
    # and uninstalls. The exact correctness of this will really depend on pip and the specific
    # libraries you are installing and uninstalling, so please use this feature with caution.
    requirement_dependency: Optional[List[str]] = None

    # System dependencies that can be installed via `apt install`. FOr example, if your photon
    # depends on `ffmpeg`, you can set `system_dependency=["ffmpeg"]`.
    system_dependency: Optional[List[str]] = None

    # The deployment template that gives a (soft) reminder to the users about how to use the
    # photon. For example, if your photon has the following:
    #   - requires gpu.a10 to run
    #   - a required env variable called ENV_A, and the user needs to set the value.
    #   - an optional env variable called ENV_B with default value "DEFAULT_B"
    #   - a required secret called SECRET_A, and the user needs to choose the secret.
    # Then, the deployment template should look like:
    #     deployment_template: Dict = {
    #       "resource_shape": "gpu.a10",
    #       "env": {
    #         "ENV_A": ENV_VAR_REQUIRED,
    #         "ENV_B": "DEFAULT_B",
    #       },
    #       "secret": [
    #         "SECRET_A",
    #       ],
    #     }
    # During photon init time, we will check the existence of the env variables and secrets,
    # issue RuntimeError if the required ones are not set, and set default values for non-existing
    # env variables that have default values.
    deployment_template: Dict[str, Any] = {
        "resource_shape": None,
        "env": {},
        "secret": [],
    }

    # The maximum number of concurrent requests that the photon can handle. In default when the photon
    # concurrency is 1, all the endpoints defined by @Photon.handler is mutually exclusive, and at any
    # time only one endpoint is running. This does not include system generated endpoints such as
    # /openapi.json, /metrics, /healthz, /favicon.ico, etc.
    #
    # This parameter does not apply to any async endpoints you define. In other words, if you define
    # an endpoint like
    #   @Photon.handler
    #   async def foo(self):
    #       ...
    # then the endpoint is not subject to the photon concurrency limit. You will need to manually
    # limit the concurrency of the endpoint yourself.
    #
    # Note that, similar to the standard multithreading design pattens, the Photon class cannot guarantee
    # thread safety when handler_max_concurrency > 1. The leopard ai framework itself is thread safe, but the
    # thread safety of the methods defines in the Photon class needs to be manually guaranteed by the
    # author of the photon.
    handler_max_concurrency: int = 1

    # The default timeout in seconds before the handler returns a timeout error.
    # Note that this does not actually kill the running user-defined function unless
    # the user defined function implements the cancellation logic.
    handler_timeout: int = 600

    # The docker base image to use for the photon. In default, we encourage you to use the
    # default base image, which provides a blazing fast loading time when running photons
    # remotely. On top of the default image, you can then install any additional dependencies
    # via `requirement_dependency` or `system_dependency`.
    image: str = BASE_IMAGE

    # The args for the base image.
    args: List[str] = BASE_IMAGE_ARGS
    cmd: Optional[List[str]] = BASE_IMAGE_CMD
    exposed_port: int = DEFAULT_PORT

    # Port used for liveness check, use the same
    # port as the deployment server by default.
    health_check_liveness_tcp_port: Optional[int] = None

    # When the server is shut down, we will wait for all the ongoing requests to finish before
    # shutting down. This is the timeout for the graceful shutdown. If the timeout is
    # reached, we will force kill the server. If not set, we use the default setting in
    # leopardai.config.DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN. If set, this parameter overrides the
    # default setting.
    timeout_graceful_shutdown: Optional[int] = None

    # During some deployment environments, the server might run behind a load balancer, and during
    # the shutdown time, the load balancer will send a SIGTERM to uvicorn to shut down the server.
    # The default behavior of uvicorn is to immediately stop receiving new traffic, and it is problematic
    # when the load balancer need to wait for some time to propagate the TERMINATING status to
    # other components of the distributed system. This parameter controls the grace period before
    # uvicorn rejects incoming traffic on SIGTERM. If not set, we use the default setting in
    # leopardai.config.DEFAULT_INCOMING_TRAFFIC_GRACE_PERIOD.
    incoming_traffic_grace_period: Optional[int] = None

    # The git repository to check out as part of the photon deployment phase.
    vcs_url: Optional[str] = None

    # internal variable to guard against accidental override of the __init__ function.
    # Sometimes, out of habit, users might override the __init__ function instead of writing the
    # init() function for the Photon class. In this case, we will issue a warning to the user
    # to remind them to use the init() function instead.
    __photon_class_constructor_called = False

    def __init__(self, name=None, model=None):
        """
        Initializes a Photon.
        """
        if name is None:
            name = self.__class__.__qualname__
        if model is None:
            model = self.__class__.__qualname__
        super().__init__(name=name, model=model)
        self._init_called = False
        self._init_res = None

        # TODO(Yangqing): add sanity check to see if the user has set handler_max_concurrency too high to
        # be handled by the default anyio number of threads.
        self._handler_semaphore: anyio.Semaphore = anyio.Semaphore(
            self.handler_max_concurrency
        )

        self.__photon_class_constructor_called = True

    @classmethod
    def _gather_routes(cls):
        def update_routes(old_, new_):
            for path, routes in new_.items():
                for method, route in routes.items():
                    old_.setdefault(path, {})[method] = route

        res = {}

        for base in cls._iter_ancestors():
            base_routes = {}
            for attr_name in dir(base):
                attr_val = getattr(base, attr_name)
                if hasattr(attr_val, PHOTON_HANDLER_PARAMS_KEY) and callable(attr_val):
                    path, method, func, kwargs = getattr(
                        attr_val, PHOTON_HANDLER_PARAMS_KEY
                    )
                    base_routes.setdefault(path, {})[method] = (func, kwargs)
            update_routes(res, base_routes)

        return res

    @classmethod
    def _iter_ancestors(cls) -> Iterator[Type["Photon"]]:
        yield cls
        for base in cls.__bases__:
            if base == Photon:
                # We still yield the Photon class, in case in the future, we add
                # default paths etc. in the Photon class.
                yield base
            elif not issubclass(base, Photon):
                # We do not yield non-Photon base classes, and any base class of
                # the Photon class (such as BasePhoton)
                continue
            else:
                yield from base._iter_ancestors()

    @property
    def _requirement_dependency(self) -> List[str]:
        deps = []
        # We add dependencies from ancestor classes to derived classes
        # and keep the order. Because we now support installation and uninstallation,
        # we do not remove redundant dependencies automatically.
        for base in reversed(list(self._iter_ancestors())):
            if base.requirement_dependency:
                deps.extend(base.requirement_dependency)
        # Do not sort or uniq pip deps lines, as order matters
        return deps

    @property
    def _system_dependency(self) -> List[str]:
        deps = OrderedDict()
        for base in reversed(list(self._iter_ancestors())):
            if base.system_dependency:
                deps.update({dep: None for dep in base.system_dependency})
        # NB: maybe we can sort and uniq system deps lines
        return list(deps.keys())

    @property
    def _deployment_template(self) -> Dict[str, Any]:
        # Verify and return the deployment template.
        if self.deployment_template is None:
            return {}
        # doing sanity check for the fields
        sanity_checked_fields = ["resource_shape", "env", "secret"]
        if any(
            field not in sanity_checked_fields
            for field in self.deployment_template.keys()
        ):
            raise ValueError(
                "Deployment template encountered a field that is not supported."
                f" Supported fields are: {sanity_checked_fields}."
            )
        # doing sanity check for the values
        resource_shape = self.deployment_template.get("resource_shape")
        if resource_shape is not None:
            if not isinstance(resource_shape, str):
                raise ValueError(
                    "Deployment template resource_shape must be a string. Found"
                    f" {resource_shape} instead."
                )
            if resource_shape not in VALID_SHAPES:
                # For now, only issue a warning if the user specified a non-standard
                # shape, and not an error. This is because we want to allow future versions
                # of the CLI to support more shapes, and we do not want to break the
                # compatibility.
                warnings.warn(
                    "Deployment template resource_shape"
                    f" {resource_shape} is not one of the"
                    " standard shapes. Just a kind reminder."
                )
        env = self.deployment_template.get("env", {})
        if not isinstance(env, dict):
            raise ValueError(
                f"Deployment template envs must be a dict. Found {env} instead."
            )
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError(
                    "Deployment template envs keys/values must be strings. Found"
                    f" {key}:{value} instead."
                )
            # Check if key is a legal env variable name.
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(
                    f"Deployment template envs key {key} is not a valid env variable"
                    " name."
                )
        secret = self.deployment_template.get("secret", [])
        if not isinstance(secret, list):
            raise ValueError(
                f"Deployment template secrets must be a list. Found {secret} instead."
            )
        for key in secret:
            if not isinstance(key, str):
                raise ValueError(
                    "Deployment template secrets must be a list of strings. Found"
                    f"invalid secret name: {key}."
                )
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(
                    f"Deployment template secrets key {key} is not a valid env variable"
                    " name."
                )
        return self.deployment_template

    @property
    def metadata(self):
        res = super().metadata

        # bookkeeping for debugging purposes: check cloudpickle and pydantic version
        # for creation time and run time sanity check
        res["cloudpickle_version"] = cloudpickle.__version__
        res["pydantic_version"] = pydantic.__version__

        res["openapi_schema"] = self._create_app(load_mount=False).openapi()

        res["py_obj"] = {
            "name": self.__class__.__qualname__,
            "obj_pkl_file": self.obj_pkl_filename,
            "py_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }

        try:
            src_file = inspect.getfile(self.__class__)
        except Exception as e:
            res["py_obj"]["src_file"] = None  # type: ignore
            res["py_obj"]["src_file_error"] = str(e)
        else:
            res["py_obj"]["src_file"] = src_file

        res.update({"requirement_dependency": self._requirement_dependency})
        res.update({"system_dependency": self._system_dependency})
        res.update({METADATA_VCS_URL_KEY: self.vcs_url})

        res.update({"deployment_template": self._deployment_template})

        res.update({
            "image": self.image,
            "args": self.args,
            "exposed_port": self.exposed_port,
        })

        if self.health_check_liveness_tcp_port is not None:
            res["health_check_liveness_tcp_port"] = self.health_check_liveness_tcp_port

        if self.cmd is not None:
            res["cmd"] = self.cmd

        return res

    def save(self, path: Optional[str] = None):
        path = super().save(path=path)
        with zipfile.ZipFile(path, "a") as photon_file:
            with photon_file.open(self.obj_pkl_filename, "w") as obj_pkl_file:
                pickler = cloudpickle.CloudPickler(obj_pkl_file, protocol=4)

                def pickler_dump(obj):
                    # internal logger opens keeps the log file opened
                    # in append mode, which is not supported to
                    # pickle, so needs to close it first before
                    # pickling
                    internal_logging.disable()
                    pickler.dump(obj)
                    internal_logging.enable()

                try:
                    from cloudpickle.cloudpickle import _extract_code_globals

                    orig_function_getnewargs = pickler._function_getnewargs
                except (ImportError, AttributeError):
                    pickler_dump(self)
                else:

                    def _function_getnewargs(func):
                        try:
                            g_names = _extract_code_globals(func.__code__)
                            for name in ["__file__", "__path__"]:
                                if name in g_names:
                                    warnings.warn(
                                        f"function {func} has used global variable"
                                        f" '{name}', its value"
                                        f" '{func.__globals__[name]}' is resolved"
                                        " during Photon creation instead of Deployment"
                                        " runtime, which may cause unexpected"
                                        " behavior."
                                    )
                        except Exception:
                            pass
                        return orig_function_getnewargs(func)

                    # We normally use loguru to do logging, in order
                    # to verify the warning message is working
                    # properly, we use assertWarns in unittest, which
                    # requires the warning message to be emiited by
                    # the python warnings module. So we need to patch
                    # the warning module to emit the warning message
                    # by warnings module but printed by loguru
                    showwarning_ = warnings.showwarning

                    def showwarning(message, *args, **kwargs):
                        logger.warning(message)
                        showwarning_(message, *args, **kwargs)

                    with patch(warnings, "showwarning", showwarning):
                        with patch(
                            pickler, "_function_getnewargs", _function_getnewargs
                        ):
                            pickler_dump(self)

            src_str = None
            try:
                src_file = inspect.getfile(self.__class__)
            except Exception:
                pass
            else:
                if os.path.exists(src_file):
                    with open(src_file, "rb") as src_file_in:
                        src_str = src_file_in.read()
            if src_str is None:
                try:
                    src_str = inspect.getsource(self.__class__)
                except Exception:
                    pass
            else:
                with photon_file.open(self.py_src_filename, "w") as src_file_out:
                    src_file_out.write(src_str)
        return path

    @classmethod
    def load(cls, photon_file, metadata) -> "Photon":
        py_version = metadata["py_obj"].get("py_version")
        cur_py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if py_version is not None and py_version != cur_py_version:
            logger.warning(
                f"Photon was created with Python {py_version} but now run with Python"
                f" {cur_py_version}, which may cause unexpected behavior."
            )

        def check_version(metadata_version, runtime_version, package: str):
            if metadata_version:
                if metadata_version != runtime_version:
                    logger.warning(
                        f"Photon was created with {package} {metadata_version} but now"
                        f" run with {package} {runtime_version}, which may cause"
                        " unexpected behavior if the versions are not compatible."
                    )
            else:
                logger.warning(
                    f"Photon was created without {package} version information, and now"
                    f" run with {package} {runtime_version}. If the versions are"
                    " not compatible, it may cause unexpected behavior, but we cannot"
                    " verify the compatibility."
                )

        check_version(
            metadata.get("cloudpickle_version"), cloudpickle.__version__, "cloudpickle"
        )
        check_version(
            metadata.get("pydantic_version"), pydantic.__version__, "pydantic"
        )

        obj_pkl_filename = metadata["py_obj"]["obj_pkl_file"]
        with photon_file.open(obj_pkl_filename) as obj_pkl_file:
            py_obj = cloudpickle.loads(obj_pkl_file.read())
        return py_obj

    def init(self):
        """
        The explicit init function that your derived Photon class should implement.
        This function is called when we create a deployment from a photon, and is
        guaranteed to run before the first api call served by the photon.
        """
        ####################################
        # Implementation note: if you are modifying Photon.init() itself, make sure that
        # the function is idempotent. In other words, if the user calls init() multiple
        # times, the function should have the same effect as calling it once. This is
        # because although we guard the init function with _call_init_once() in the
        # Photon framework itself, the user might stil be explicitly calling it multiple
        # times, and we do not want side effect to happen.
        ####################################
        # sets the envs and secrets specified in the deployment template. For envs, we
        # set the default values if the env is not set, which helps local deployment and
        # debugging. For secret, we explicitly require the user to set it before running
        # the photon (on the platform, secret is filled by the platform if specified).
        envs = self._deployment_template.get("env", {})
        for key in envs:
            if os.environ.get(key) is None:
                if envs[key] == ENV_VAR_REQUIRED:
                    warnings.warn(
                        f"This photon expects env variable {key}, but it's not set."
                        " Please set it before running the photon, or you may get"
                        " unexpected behavior.",
                        RuntimeWarning,
                    )
                else:
                    os.environ[key] = envs[key]
        secrets = self._deployment_template.get("secret", [])
        for s in secrets:
            if os.environ.get(s) is None:
                warnings.warn(
                    f"This photon expects secret {s}, but it's not set."
                    " Please set it before running the photon, or you may get"
                    " unexpected behavior.",
                    RuntimeWarning,
                )

    def _call_init_once(self):
        """
        Internal function that calls the init function once.
        """
        if not self.__photon_class_constructor_called:
            raise RuntimeError(
                "It seems that your photon class has overridden the __init__ function."
                " Did you accidentally write `def __init__(self)` instead of `def"
                " init(self)`, aka no underscore? The latter is the correct way to"
                " define the init function for your photon's application side logic. If"
                " you indeed mean to override the __init__ function, please make sure"
                " that inside the __init__ function, you call the super().__init__"
                " function to ensure the correct initialization of the Photon class."
            )
        if not self._init_called:
            with _photon_initialize_lock:
                # acquire the lock, and check again.
                if self._init_called:
                    return
                else:
                    self._init_called = True
                    # run Photon's init function.
                    Photon.init(self)
                    # run the user-defined init function
                    self._init_res = self.init()
        return self._init_res

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _add_cors_middlewares(app):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOW_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _add_auth_middleware_if_needed(self, app):
        local_token = copy.deepcopy(get_local_deployment_token())
        if not local_token:
            # no need to add auth middleware.
            return
        else:
            from starlette.middleware.base import BaseHTTPMiddleware

            class AuthMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request: Request, call_next):
                    authorization: Optional[str] = request.headers.get("Authorization")
                    token = authorization.split(" ")[-1] if authorization else ""
                    if token != local_token:
                        return JSONResponse(
                            status_code=status.HTTP_403_FORBIDDEN,
                            content={"detail": "Invalid or expired token"},
                        )
                    else:
                        return await call_next(request)

            app.add_middleware(AuthMiddleware)

    def _replace_openapi_if_needed(self, app):
        # If the class has a `openapi` method, we replace the default openapi
        # schema with the one returned by the `openapi` method.
        if hasattr(self, "openapi"):
            logger.debug(
                "Replacing the default openapi schema with the one returned by the"
                " `openapi` method. Note: you are responsible yourself to ensure"
                " that the schema returned by the `openapi` method is a valid"
                "  Callable[[], Dict[str, Any]] type."
            )
            app.openapi = self.openapi()  # type: ignore

    def _create_app(self, load_mount):
        title = self._photon_name.replace(".", "_")
        app = FastAPI(
            title=title,
            description=(
                self.__doc__
                if self.__doc__
                else f"leopard AI Photon API {self._photon_name}"
            ),
        )

        # web hosted cdn and inference api have different domains:
        # https://github.com/leopardai/leopard/issues/358
        # TODO: remove this once our Ingress is capable of handling all these
        # and make all communication internal from the deployment point of view
        self._add_cors_middlewares(app)
        # Add the authentication middleware if local token is specified.
        self._add_auth_middleware_if_needed(app)

        self._register_routes(app, load_mount)
        self._collect_metrics(app)

        return app