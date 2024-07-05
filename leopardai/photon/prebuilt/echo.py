from leopardai.photon import Photon


class Echo(Photon):
    """
    A very simple example photon for demo and debugging purposes. It exposes one
    endpoint `/echo` that returns the input string as the response. If you are
    using the leopard CLI, you can run this photon with the following command:

    ```bash
    leopard photon run -n echo -m leopardai.photon.prebuilt.echo.Echo
    ```

    And you can use the client to call the endpoint:

    ```python
    client.echo(input="Hello World!")
    ```
    """

    @Photon.handler
    def echo(self, input: str) -> str:
        """
        Echo the input string.
        """
        return input