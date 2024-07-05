from leopardai.photon import Photon


class Foo(Photon):
    image = "leopardai/base:latest"

    def init(self):
        pass

    @Photon.handler
    def foo(self):
        return