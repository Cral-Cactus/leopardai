import os
import tempfile

# Set cache dir to a temp dir before importing anything from leopardai
tmpdir = tempfile.mkdtemp()
os.environ["leopard_CACHE_DIR"] = tmpdir

import unittest

from leopardai._internal.logging import log as internal_log
from leopardai._internal.logging import _LOGFILE_BASE, enable, disable


class TestInternalLog(unittest.TestCase):
    def setUp(self):
        os.environ["leopard_ENABLE_INTERNAL_LOG"] = "1"
        enable()

    def tearDown(self):
        disable()
        del os.environ["leopard_ENABLE_INTERNAL_LOG"]

    def test_log(self):
        msg = "some random message"
        internal_log(msg)
        self.assertTrue(os.path.exists(_LOGFILE_BASE))
        with open(_LOGFILE_BASE, "r") as f:
            self.assertIn(msg, f.read())


if __name__ == "__main__":
    unittest.main()