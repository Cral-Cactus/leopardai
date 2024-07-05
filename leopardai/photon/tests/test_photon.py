import os
import tempfile
import time
import warnings

# Set cache dir to a temp dir before importing anything from leopardai
tmpdir = tempfile.mkdtemp()
os.environ["leopard_CACHE_DIR"] = tmpdir

try:
    from flask import Flask
except ImportError:
    has_flask = False
else:
    has_flask = True

try:
    import gradio as gr
except ImportError:
    has_gradio = False
else:
    has_gradio = True

try:
    import hnsqlite  # noqa: F401
except ImportError:
    has_hnsqlite = False
else:
    has_hnsqlite = True

try:
    from asgi_proxy import asgi_proxy
except ImportError:
    has_asgi_proxy = False
else:
    has_asgi_proxy = True

import asyncio
import concurrent.futures
from io import BytesIO
import inspect
from textwrap import dedent
import shutil
import subprocess
import threading
import sys
from typing import Dict, List
import unittest
import zipfile

from fastapi import FastAPI
import requests
import torch

import leopardai
from leopardai import Client
from leopardai.config import ENV_VAR_REQUIRED
from leopardai.photon.constants import METADATA_VCS_URL_KEY
from leopardai.photon import Photon, HTTPException, PNGResponse, FileParam, StaticFiles
from leopardai.photon.util import (
    create as create_photon,
    load_metadata,
)
from leopardai.util import switch_cwd, find_available_port
from utils import random_name, photon_run_local_server, skip_if_macos


class CustomPhoton(Photon):
    input_example = {"x": 2.0}

    def init(self):
        self.nn = torch.nn.Linear(1, 1)

    @Photon.handler("some_path", example=input_example)
    def run(self, x: float) -> float:
        return self.nn(torch.tensor(x).reshape(1, 1)).item()

    @Photon.handler("some_path_2")
    def run2(self, x: float) -> float:
        return x * 2

    @Photon.handler("")
    def run3(self, x: float) -> float:
        return x * 3


class CustomPhotonWithDepTemplate(Photon):
    deployment_template: Dict = {
        "resource_shape": "gpu.a10",
        "env": {
            "leopard_FOR_TEST_ENV_A": ENV_VAR_REQUIRED,
            "leopard_FOR_TEST_ENV_B": "DEFAULT_B",
        },
        "secret": ["leopard_FOR_TEST_SECRET_A"],
    }


class CustomPhotonWithInvalidDepTemplateEnv(Photon):
    deployment_template: Dict = {
        "resource_shape": "gpu.a10",
        "env": {
            # Note: intentional single quote simulating user input error
            "leopard_FOR_TEST_ENV_A'": ENV_VAR_REQUIRED,
            "leopard_FOR_TEST_ENV_B": "DEFAULT_B",
        },
        "secret": ["leopard_FOR_TEST_SECRET_A"],
    }


class CustomPhotonWithInvalidDepTemplateSecret(Photon):
    deployment_template: Dict = {
        "resource_shape": "gpu.a10",
        "env": {
            "leopard_FOR_TEST_ENV_A": ENV_VAR_REQUIRED,
            "leopard_FOR_TEST_ENV_B": "DEFAULT_B",
        },
        # Note: intentional single quote simulating user input error
        "secret": ["leopard_FOR_TEST_SECRET_A'"],
    }


class CustomPhotonWithCustomDeps(Photon):
    requirement_dependency = ["torch"]
    system_dependency = ["ffmpeg"]


test_txt = tempfile.NamedTemporaryFile(suffix=".txt")
with open(test_txt.name, "w") as f:
    for i in range(10):
        f.write(f"line {i}\n")
    f.flush()


class CustomPhotonWithCustomExtraFiles(Photon):
    extra_files = {
        "test.txt": test_txt.name,
        "a/b/c/test.txt": test_txt.name,
    }

    def init(self):
        with open("test.txt") as f:
            self.lines = f.readlines()
        with open("a/b/c/test.txt") as f:
            self.lines.extend(f.readlines())

    @Photon.handler("line")
    def run(self, n: int) -> str:
        if n >= len(self.lines):
            raise HTTPException(
                status_code=400, detail=f"n={n} exceeds total #lines ({self.lines})"
            )
        return self.lines[n]


class CustomPhotonWithPNGResponse(Photon):
    @Photon.handler()
    def img(self, content: str) -> PNGResponse:
        img_io = BytesIO()
        img_io.write(content.encode("utf-8"))
        img_io.seek(0)
        return PNGResponse(img_io)


class CustomPhotonWithMount(Photon):
    @Photon.handler(mount=True)
    def myapp(self):
        app = FastAPI()

        @app.post("/hello")
        def hello():
            return "world"

        return app

    @Photon.handler()
    def run(self):
        return "hello"


class ChildPhoton(Photon):
    @Photon.handler()
    def greet(self) -> str:
        return "hello from child"


class ParentPhoton(Photon):
    @Photon.handler()
    def greet(self) -> str:
        return "hello from parent"

    @Photon.handler(mount=True)
    def child(self):
        return ChildPhoton()


class TestPhoton(unittest.TestCase):
    def setUp(self):
        # pytest imports test files as top-level module which becomes
        # unavailable in server process
        if "PYTEST_CURRENT_TEST" in os.environ:
            import cloudpickle

            cloudpickle.register_pickle_by_value(sys.modules[__name__])

    def test_deployment_template(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        self.assertEqual(
            ph.deployment_template, {"resource_shape": None, "env": {}, "secret": []}
        )

        if "leopard_FOR_TEST_ENV_A" in os.environ:
            del os.environ["leopard_FOR_TEST_ENV_A"]

        try:
            ph = CustomPhotonWithDepTemplate(name=name)
        except Exception as e:
            self.fail(
                "Although env is missing, creating a photon should not fail."
                f" Details: {e}"
            )

        ph = CustomPhotonWithDepTemplate(name=name)
        with self.assertWarnsRegex(RuntimeWarning, ".*leopard_FOR_TEST_ENV_A.*"):
            ph._call_init_once()

        os.environ["leopard_FOR_TEST_ENV_A"] = "value_a"
        ph = CustomPhotonWithDepTemplate(name=name)
        with self.assertWarnsRegex(RuntimeWarning, ".*leopard_FOR_TEST_SECRET_A.*"):
            ph._call_init_once()

        os.environ["leopard_FOR_TEST_SECRET_A"] = "value_a"
        ph = CustomPhotonWithDepTemplate(name=name)
        with warnings.catch_warnings():
            ph._call_init_once()
        self.assertEqual(os.environ["leopard_FOR_TEST_ENV_B"], "DEFAULT_B")
        self.assertEqual(os.environ["leopard_FOR_TEST_ENV_A"], "value_a")
        self.assertEqual(os.environ["leopard_FOR_TEST_SECRET_A"], "value_a")

        metadata = ph.metadata

        self.assertIn("deployment_template", metadata)
        self.assertEqual(
            metadata["deployment_template"],
            {
                "resource_shape": "gpu.a10",
                "env": {
                    "leopard_FOR_TEST_ENV_A": ENV_VAR_REQUIRED,
                    "leopard_FOR_TEST_ENV_B": "DEFAULT_B",
                },
                "secret": ["leopard_FOR_TEST_SECRET_A"],
            },
        )

    def test_deployment_template_invalid_env_or_secret(self):
        name = random_name()
        ph = CustomPhotonWithInvalidDepTemplateEnv(name=name)
        with self.assertRaisesRegex(ValueError, "leopard_FOR_TEST_ENV_A'"):
            ph._deployment_template()

        ph = CustomPhotonWithInvalidDepTemplateSecret(name=name)
        with self.assertRaisesRegex(ValueError, "leopard_FOR_TEST_SECRET_A'"):
            ph._deployment_template()

    def test_run(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        x = 2.0
        y1 = ph.run(x)

        xtensor = torch.tensor(x).reshape(1, 1)
        y2 = ph.nn(xtensor).item()
        self.assertEqual(y1, y2)

    def test_save_load(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        x = 2.0
        y1 = ph.run(x)

        path = ph.save()

        ph = leopardai.photon.load(path)
        y2 = ph.run(x)
        self.assertEqual(y1, y2)

    def test_run_server(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        path = ph.save()

        proc, port = photon_run_local_server(path=path)

        x = 2.0
        res = requests.post(
            f"http://localhost:{port}/some_path",
            json={"x": x},
        )
        self.assertEqual(res.status_code, 200)
        res = requests.post(
            f"http://localhost:{port}",
            json={"x": x},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), 6.0)
        proc.kill()

    def test_client(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        x = 2.0
        y1 = ph.run(x)
        path = ph.save()

        proc, port = photon_run_local_server(path=path)
        url = f"http://localhost:{port}"

        client = Client(url)
        y2 = client.some_path(x=x)
        self.assertEqual(y1, y2)
        try:
            client.some_path_does_not_exist(x=x)
        except AttributeError:
            pass
        else:
            self.fail("AttributeError not raised")
        proc.kill()

    def test_ph_cli(self):
        with tempfile.NamedTemporaryFile(suffix=".py") as f:
            f.write(dedent("""
from leopardai.photon import Photon


class Counter(Photon):
    def init(self):
        self.counter = 0

    @Photon.handler("add")
    def add(self, x: int) -> int:
        self.counter += x
        return self.counter

    @Photon.handler("sub")
    def sub(self, x: int) -> int:
        self.counter -= x
        return self.counter
""").encode("utf-8"))
            f.flush()
            for model in [f"py:{f.name}:Counter", f"{f.name}:Counter"]:
                proc, port = photon_run_local_server(name="counter", model=model)
                res = requests.post(
                    f"http://127.0.0.1:{port}/add",
                    json={"x": 1},
                )
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json(), 1)

                res = requests.post(
                    f"http://127.0.0.1:{port}/add",
                    json={"x": 1},
                )
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json(), 2)

                res = requests.post(
                    f"http://127.0.0.1:{port}/sub",
                    json={"x": 2},
                )
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json(), 0)
                proc.kill()

    def test_photon_file_metadata(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        path = ph.save()
        metadata = load_metadata(path)
        self.assertEqual(metadata["name"], name)
        self.assertEqual(metadata["model"], "CustomPhoton")
        self.assertTrue("image" in metadata)
        self.assertTrue("args" in metadata)
        self.assertFalse("cmd" in metadata)  # no cmd is specified to indicates default
        self.assertTrue("exposed_port" in metadata)

        # check for openapi schema
        self.assertTrue("openapi_schema" in metadata)
        self.assertTrue("/some_path" in metadata["openapi_schema"]["paths"])
        # check for annotated example
        self.assertEqual(
            metadata["openapi_schema"]["paths"]["/some_path"]["post"]["requestBody"][
                "content"
            ]["application/json"]["example"],
            CustomPhoton.input_example,
        )
        # handler without specifying example should not have 'example' in metadata
        with self.assertRaises(KeyError) as raises:
            metadata["openapi_schema"]["paths"]["/some_path_2"]["post"]["requestBody"][
                "content"
            ]["application/json"]["example"]
        self.assertEqual(raises.exception.args[0], "example")

        self.assertEqual(len(metadata["requirement_dependency"]), 0)

        self.assertEqual(
            metadata["py_obj"]["py_version"],
            f"{sys.version_info.major}.{sys.version_info.minor}",
        )

    def test_liveness_check(self):
        class LivenessCheckPhoton(Photon):
            pass

        ph = LivenessCheckPhoton(name=random_name())
        path = ph.save()
        proc, port = photon_run_local_server(path=path)
        res = requests.get(f"http://localhost:{port}/livez")
        proc.kill()
        self.assertEqual(res.status_code, 200, res.text)

        liveness_port = find_available_port()

        class CustomLivenessCheckPhoton(Photon):
            health_check_liveness_tcp_port = liveness_port

        ph = CustomLivenessCheckPhoton(name=random_name())
        path = ph.save()
        metadata = load_metadata(path)
        self.assertEqual(metadata["health_check_liveness_tcp_port"], liveness_port)
        proc, port = photon_run_local_server(path=path)
        res = requests.get(f"http://localhost:{liveness_port}/livez")
        proc.kill()
        self.assertEqual(res.status_code, 200)

    def test_custom_image_photon_metadata(self):
        class CustomImage(Photon):
            image = "a:b"
            exposed_port = 8765
            cmd = ["python", "-m", "http.server", str(exposed_port)]

        ph = CustomImage(name=random_name())
        path = ph.save()
        metadata = load_metadata(path)
        self.assertEqual(metadata["image"], CustomImage.image)
        self.assertEqual(metadata["exposed_port"], CustomImage.exposed_port)
        self.assertEqual(metadata["cmd"], CustomImage.cmd)

    def test_custom_dependency(self):
        name = random_name()
        ph = CustomPhotonWithCustomDeps(name=name)
        path = ph.save()
        metadata = load_metadata(path)
        self.assertEqual(
            metadata["requirement_dependency"],
            CustomPhotonWithCustomDeps.requirement_dependency,
        )
        self.assertEqual(
            metadata["system_dependency"],
            CustomPhotonWithCustomDeps.system_dependency,
        )

    def test_metrics(self):
        name = random_name()
        ph = CustomPhoton(name=name)
        path = ph.save()

        proc, port = photon_run_local_server(path=path)

        for x in range(5):
            res = requests.post(
                f"http://127.0.0.1:{port}/some_path",
                json={"x": float(x)},
            )
            self.assertEqual(res.status_code, 200)
        res = requests.get(f"http://127.0.0.1:{port}/metrics")
        self.assertEqual(res.status_code, 200)

        # prometheus-fastapi-instrumentator>=6.1.0 added
        # "method" label to "http_request_duration_seconds" metrics
        self.assertRegex(
            res.text,
            r'http_request_duration_seconds_count{handler="/some_path"(,method="POST")?}',
        )
        self.assertRegex(
            res.text,
            r'http_request_duration_seconds_bucket{handler="/some_path",le="0.01"(,method="POST")?}',
        )
        self.assertRegex(
            res.text,
            r'http_request_duration_seconds_bucket{handler="/some_path",le="0.78"(,method="POST")?}',
        )
        self.assertRegex(
            res.text,
            r'http_request_duration_seconds_bucket{handler="/some_path",le="1.1"(,method="POST")?}',
        )
        self.assertRegex(
            res.text,
            r'http_request_duration_seconds_bucket{handler="/some_path",le="2.3"(,method="POST")?}',
        )
        proc.kill()

    def test_extra_files(self):
        name = random_name()
        ph = CustomPhotonWithCustomExtraFiles(name=name)
        path = ph.save()

        proc, port = photon_run_local_server(path=path)
        res = requests.post(
            f"http://127.0.0.1:{port}/line",
            json={"n": 1},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), "line 1\n")
        proc.kill()

        temp_dir = tempfile.mkdtemp(dir=tmpdir)
        with switch_cwd(temp_dir):
            sub_dir = os.path.join(temp_dir, "repo")
            os.makedirs(sub_dir)

            layer_dir = sub_dir
            for layer in range(3):
                layer_dir = os.path.join(layer_dir, str(layer))
                os.makedirs(layer_dir)
                with open(os.path.join(layer_dir, "a.py"), "w") as f:
                    f.write(str(layer))

            dot_file = os.path.join(temp_dir, "hidden", ".abc.conf")
            os.makedirs(os.path.dirname(dot_file))
            with open(dot_file, "w") as f:
                f.write("abc")
            dot_file_mode = os.stat(dot_file).st_mode

            class RecursiveIncludePhoton(Photon):
                extra_files = [os.path.relpath(sub_dir, temp_dir)]

                @Photon.handler(method="GET")
                def cat(self, path: str) -> str:
                    with open(path) as f:
                        return f.read()

                @Photon.handler(method="GET")
                def mask(self, path: str) -> int:
                    return os.stat(path).st_mode

            ph = RecursiveIncludePhoton(name=random_name())
            path = ph.save()
            proc, port = photon_run_local_server(path=path)

            res = requests.get(
                f"http://localhost:{port}/cat", params={"path": "repo/0/a.py"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), "0")

            res = requests.get(
                f"http://localhost:{port}/cat", params={"path": "repo/0/1/a.py"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), "1")

            res = requests.get(
                f"http://localhost:{port}/cat", params={"path": "repo/0/1/2/a.py"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), "2")

            res = requests.get(
                f"http://localhost:{port}/cat", params={"path": "hidden/.abc.conf"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json(), "abc")

            res = requests.get(
                f"http://localhost:{port}/mask", params={"path": "hidden/.abc.conf"}
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(
                res.json(), dot_file_mode, f"{oct(res.json())} != {oct(dot_file_mode)}"
            )
