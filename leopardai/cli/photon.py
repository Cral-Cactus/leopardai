from datetime import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Optional, List, Tuple

from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
import click

from loguru import logger

from leopardai.api.v1.workspace_record import WorkspaceRecord
from leopardai.api.connection import Connection
from leopardai.api import photon as api
from leopardai.api import types
from leopardai.api.workspace import WorkspaceInfoLocalRecord
from leopardai.api.secret import create_secret, list_secret
from leopardai import config
from leopardai.photon import util as photon_util
from leopardai.photon import Photon
from leopardai.photon.base import (
    find_all_local_photons,
    find_local_photon,
    remove_local_photon,
)
from leopardai.photon.constants import METADATA_VCS_URL_KEY
from leopardai.photon.download import fetch_code_from_vcs
from leopardai.util import find_available_port

from .util import (
    click_group,
    guard_api,
    check,
    explain_response,
)
from leopardai.api.v1.client import APIClient
from leopardai.api.v1.types.common import Metadata
from leopardai.api.v1.types.deployment import leopardDeployment
from leopardai.api.v1.types.deployment_operator_v1alpha1.affinity import (
    leopardResourceAffinity,
)
from leopardai.api.v1.types.deployment_operator_v1alpha1.deployment import (
    leopardDeploymentUserSpec,
    ResourceRequirement,
    AutoScaler,
    ScaleDown,
    HealthCheck,
    HealthCheckLiveness,
)
from leopardai.api.v1.photon import make_mounts_from_strings, make_env_vars_from_strings
from leopardai.api.v1.deployment import make_token_vars_from_config
from leopardai.config import ENV_VAR_REQUIRED
import warnings

console = Console(highlight=False)


DEFAULT_RESOURCE_SHAPE = "cpu.small"


def _get_ordered_photon_ids_or_none(
    name: str, public_photon: bool
) -> Optional[List[str]]:
    """Returns a list of photon ids for a given name, in the order newest to
    oldest. If no photon of such name exists, returns None.
    """

    client = APIClient()

    photons = client.photon.list_all(public_photon=public_photon)

    target_photons = [p for p in photons if p.name == name]  # type: ignore
    if len(target_photons) == 0:
        return None
    target_photons.sort(key=lambda p: p.created_at, reverse=True)
    return [p.id_ for p in target_photons]


def _get_most_recent_photon_id_or_none(name: str, public_photon: bool) -> Optional[str]:
    """Returns the most recent photon id for a given name. If no photon of such
    name exists, returns None.
    """
    photon_ids = _get_ordered_photon_ids_or_none(name, public_photon)
    return photon_ids[0] if photon_ids else None


def _create_workspace_token_secret_var_if_not_existing(conn: Connection):
    """
    Adds the workspace token as a secret environment variable.
    """
    workspace_token = WorkspaceInfoLocalRecord.get_current_workspace_token() or ""
    existing_secrets = guard_api(
        list_secret(conn),
        msg="Failed to list secrets.",
    )
    if "leopard_WORKSPACE_TOKEN" not in existing_secrets:
        response = create_secret(conn, ["leopard_WORKSPACE_TOKEN"], [workspace_token])
        explain_response(
            response,
            None,
            "Failed to create secret leopard_WORKSPACE_TOKEN.",
            "Failed to create secret leopard_WORKSPACE_TOKEN.",
            exit_if_4xx=True,
        )


@click_group()
def photon():
    """
    Manages photons locally and on the leopard AI cloud.

    Photon is at the core of leopard AI's abstraction: it is a Python centric
    abstraction of an AI model or application, and provides a simple interface
    to specify dependencies, extra files, and environment variables. For more
    details, see `leopardai.photon.Photon`.

    The photon command is used to create photons locally, push and fetch photons
    between local and remote, and run, list and delete photons either locally or
    remotely.
    """
    pass


@photon.command()
@click.option("--name", "-n", help="Name of the scaffolding file", default="main.py")
def scaffold(name: str):
    """
    Creates a scaffolding main.py file for a new photon. The file serves as a starting
    point that you can modify to create your own implementations. After implementing
    your photon, you can use `lep photon create -n [name] -m main.py` to create a
    photon from the file.
    """
    check(name.endswith(".py"), "Scaffolding file must end with .py")
    check(
        not os.path.exists(name),
        f"File {name} already exists. Please choose another name.",
    )
    from leopardai.photon.prebuilt import template

    shutil.copyfile(template.__file__, name)
    console.print(f"Created scaffolding file [green]{name}[/].")


@photon.command()
@click.option("--name", "-n", help="Name of the photon", required=True)
@click.option("--model", "-m", help="Model spec", required=True)
@click.option(
    "--requirements",
    "-r",
    help=(
        "Path to file that contains additional requirements, such as a requirements.txt"
        " file."
    ),
    default=None,
)
def create(name, model, requirements):
    """
    Creates a new photon in the local environment.
    For specifics on the model spec, see `leopardai.photon.Photon`. To push a photon
    to the workspace, use `lep photon push`.

    Developer note: insert a link to the photon documentation here.
    """
    try:
        photon = photon_util.create(name=name, model=model)
    except Exception as e:
        console.print(f"Failed to create photon: [red]{e}[/]")
        sys.exit(1)
    if requirements:
        if not os.path.exists(requirements):
            console.print(f"Requirements file {requirements} does not exist.")
            sys.exit(1)
        with open(requirements, "r") as f:
            deps = [r.strip() for r in f.readlines()]
        logger.info(f"Adding requirements from {requirements}: {deps}")
        if isinstance(photon, Photon):
            if photon.requirement_dependency is None:
                photon.requirement_dependency = deps
            else:
                photon.requirement_dependency.extend(deps)
    try:
        photon_util.save(photon)
    except Exception as e:
        console.print(f"Failed to save photon: [red]{e}[/]")
        sys.exit(1)
    console.print(f"Photon [green]{name}[/green] created.")


@photon.command()
@click.option(
    "--name",
    "-n",
    help=(
        "Name of the photon to delete. If `--all` is specified, all versions of the"
        " photon with this name will be deleted. Otherwise, remove the latest"
        " version of the photon with this name."
    ),
)
@click.option(
    "--local", "-l", is_flag=True, help="Remove local photons instead of remote."
)
@click.option(
    "--id", "-i", "id_", help="The specific version id of the photon to remove."
)
@click.option(
    "--all", "-a", "all_", is_flag=True, help="Remove all versions of the photon."
)
@click.option(
    "--public-photon",
    is_flag=True,
    help=(
        "If specified, remove the photon from the public photon registry. Note that"
        " public photons can only be managed by leopard, so this option is hidden"
        " by default, but we provide this helpstring for documentation purposes."
    ),
    hidden=True,
    default=False,
)
def remove(name, local, id_, all_, public_photon):
    """
    Removes a photon. The behavior of this command depends on whether one has
    logged in to the leopard AI cloud via `lep login`. If one has logged in, this
    command will remove the photon from the workspace. Otherwise, or of `--local`
    is explicitly specified, it will remove the photon from the local environment.
    """
    check(
        not (name and id_), "Cannot specify both --name and --id. Use one or the other."
    )
    check(name or id_, "Must specify either --name or --id.")
    check(
        not (public_photon and local),
        "Cannot specify --public-photon and --local both.",
    )

    if not local and WorkspaceRecord.get_current_workspace_id() is not None:
        # Remove remote photon.

        client = APIClient()
        # Find ids that we need to remove
        if name:
            # Remove all versions of the photon.
            ids = _get_ordered_photon_ids_or_none(name, public_photon=public_photon)
            check(ids, f"Cannot find photon with name [yellow]{name}[/].")

            ids = [ids[0]] if (not all_) else ids  # type: ignore
        else:
            ids = [id_]
        # Actually remove the ids
        for id_to_remove in ids:  # type: ignore
            client.photon.delete(id_to_remove)
            console.print(f"Photon id [green]{id_to_remove}[/] removed.")
        return
    else:
        # local mode
        check(name, "Must specify --name when removing local photon")
        check(find_local_photon(name), f"Photon [red]{name}[/] does not exist.")
        remove_local_photon(name, remove_all=all_)
        console.print(
            f"{'' if all_ else 'Latest version of '}Photon [green]{name}[/] removed."
        )
        return


@photon.command(name="list")
@click.option("--local", "-l", help="If specified, list local photons", is_flag=True)
@click.option(
    "--pattern", help="Regular expression pattern to filter photon names", default=None
)
@click.option(
    "--public-photon",
    is_flag=True,
    help="If specified, list photons from the public photon registry.",
    default=False,
)
def list_command(local, pattern, public_photon):
    """
    Lists all photons. If one has logged in to the leopard AI cloud via `lep login`,
    this command will list all photons in the leopard AI cloud. Otherwise, or if
    `--local` is explicitly specified, it will list all photons in the local
    environment.
    """
    check(
        not (public_photon and local),
        "Cannot specify --public-photon and --local both.",
    )
    if not local and WorkspaceRecord.get_current_workspace_id() is not None:
        client = APIClient()
        photons = client.photon.list_all(public_photon=public_photon)
        # Note: created_at returned by the server is in milliseconds, and as a
        # result we need to divide by 1000 to get seconds that is understandable
        # by the Python CLI.
        records = [
            (photon.name, photon.model, photon.id_, photon.created_at / 1000)
            for photon in photons
        ]
        ws_id = client.get_workspace_id()
        ws_name = client.get_workspace_name()
        if ws_name:
            title = f"Photons in workspace {ws_id}({ws_name})"
        else:
            title = f"Photons in workspace {ws_id}"
    else:
        records = find_all_local_photons()
        records = [
            (name, model, id_, creation_time)
            for id_, name, model, _, creation_time in records
        ]
        # We use current_workspace_id = None to indicate that we are in local mode.
        ws_id = None
        title = "Local Photons"

    table = Table(title=title, show_lines=True)
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("ID")
    table.add_column("Created At")

    records_by_name = {}
    for name, model, id_, creation_time in records:
        if pattern is None or re.match(pattern, name):
            records_by_name.setdefault(name, []).append((model, id_, creation_time))

    # Sort by creation time and print
    for name, sub_records in records_by_name.items():
        sub_records.sort(key=lambda r: r[2], reverse=True)
        model_table = Table(show_header=False, box=None)
        id_table = Table(show_header=False, box=None)
        creation_table = Table(show_header=False, box=None)
        for model, id_, creation_time in sub_records:
            model_table.add_row(model)
            id_table.add_row(id_)
            # photon database stores creation time as a timestamp in
            # milliseconds, so we need to convert.
            creation_table.add_row(
                datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
            )
        table.add_row(name, model_table, id_table, creation_table)
    console.print(table)
    if ws_id:
        console.print("To show local photons, use the `--local` flag.")