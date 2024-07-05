import unittest


class TestImport(unittest.TestCase):
    def test_import_has_photon(self):
        import leopardai

        self.assertTrue(hasattr(leopardai, "photon"))

    def test_version(self):
        from leopardai import __version__

        self.assertIsNotNone(__version__)


if __name__ == "__main__":
    unittest.main()