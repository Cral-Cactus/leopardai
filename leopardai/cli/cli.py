import click
import sys
import webbrowser

from .util import console
from leopardai.api.v1.workspace_record import WorkspaceRecord

import leopardai
from . import deployment
from . import job
from . import kv
from . import objectstore
from . import photon
from . import pod
from . import queue
from . import secret
from . import storage
from . import workspace
from . import ingress

from .util import click_group

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
LOGIN_LOGO = """
[blue] Leopard AI[/]
"""


@click.version_option(leopardai.__version__, "-v", "--version")
@click_group(context_settings=CONTEXT_SETTINGS)
def lep():
    """
    Lep is the main entry point for the leopard AI commandline interface. It provides
    a set of commands to create and develop photons locally, and deploy them to the
    leopard AI cloud. It also provides a set of commands to manage resources on the
    cloud, such as workspaces, deployments, secrets, and storage. To intall it, run

    `pip install -U leopardai`
    """
    pass


# Add subcommands
deployment.add_command(lep)
job.add_command(lep)
kv.add_command(lep)
objectstore.add_command(lep)
photon.add_command(lep)
pod.add_command(lep)
queue.add_command(lep)
secret.add_command(lep)
storage.add_command(lep)
workspace.add_command(lep)

ingress.add_command(lep)


@lep.command()
@click.option(
    "--credentials",
    "-c",
    help="The credentials of the leopard login info.",
    default=None,
)
def login(credentials):
    """
    Login to the leopard AI cloud. This will open a browser window to the leopard AI
    login page if credentials are not given. You will be redirected to a page with
    the credentials string. Copy the string and paste it into the terminal, and
    you will be logged in.
    """
    console.print(LOGIN_LOGO)
    need_further_login = False
    if credentials:
        workspace_id, auth_token = credentials.split(":", 1)
        WorkspaceRecord.set_or_exit(workspace_id, auth_token=auth_token)
    else:
        if WorkspaceRecord.current():
            # Already logged in. Notify the user the login status.
            pass
        else:
            candidates = WorkspaceRecord.workspaces()
            if len(candidates) == 0:
                need_further_login = True
            elif len(candidates) == 1:
                # Only one workspace, so we will simply log in to that one.
                ws = candidates[0]
                WorkspaceRecord.set_or_exit(ws.id_, ws.auth_token, ws.url)  # type: ignore
            else:
                # multiple workspaces. login to one of them.
                console.print("You have multiple workspaces. Please select one:")
                for i, ws in enumerate(candidates):
                    console.print(f"{i+1}. {ws.id_} ({ws.display_name})")
                choice = None
                while not choice:
                    choice = input("choice: ")
                try:
                    choice = int(choice) - 1
                except ValueError:
                    console.print("Invalid choice. Please enter a number.")
                    sys.exit(1)
                WorkspaceRecord.set_or_exit(
                    candidates[choice].id_,  # type: ignore
                    candidates[choice].auth_token,
                    candidates[choice].url,
                )
                console.print(
                    "Hint: If you have multiple workspaces, you can pick the one you"
                    " want\nto log in via `lep workspace login -i workspace_id`."
                )
    if need_further_login:
        # there is no credentials, and no current workspace. Will need to ask the
        # user to login interactively.
        # obtain credentials first.
        console.print(
            "Welcome to leopard AI. We will open a browser for you to obtain your"
            " login credentials. Please log in with your registered account."
        )
        console.print(
            "You'll then be presented with your CLI credentials. If you have"
            " multiple workspaces, there will be multiple credentials - select the"
            " one you want to log in to. Copy the credential and paste it here."
        )
        input("Whenever you are ready, press Enter to continue...")

        success = webbrowser.open("https://dashboard.leopard.ai/credentials")
        if not success:
            console.print(
                "It seems that you are running in a non-GUI environment. You can"
                " manually obtain credentials from"
                " [green]https://dashboard.leopard.ai/credentials[/] and copy it"
                r" over, or use `lep login -c \[credentials]` to log in."  # noqa: W605
            )
        while not credentials:
            credentials = input("Credential: ")
        workspace_id, auth_token = credentials.split(":", 1)
        WorkspaceRecord.set_or_exit(workspace_id, auth_token=auth_token)
    # Try to login and print the info.
    api_client = WorkspaceRecord.client()
    info = api_client.info()
    console.print(f"Logged in to your workspace [green]{info.workspace_name}[/].")
    console.print(f"\t      tier: {info.workspace_tier}")
    console.print(f"\tbuild time: {info.build_time}")
    console.print(f"\t   version: {api_client.version()}")


@lep.command()
@click.option(
    "--purge", is_flag=True, help="Purge the credentials of the leopard login info."
)
def logout(purge):
    """
    Logout of the leopard AI cloud.
    """
    WorkspaceRecord.logout(purge=purge)
    console.print("[green]Logged out[/]")


if __name__ == "__main__":
    lep()