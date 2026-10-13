# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Framework discovery must not require unrelated models for Ego4WAM."""

import unittest
from types import SimpleNamespace
from unittest import mock

from starVLA.model.framework import base_framework


class FrameworkDiscoveryTest(unittest.TestCase):
    def _build(self, name):
        constructor = mock.Mock()
        cfg = SimpleNamespace(framework=SimpleNamespace(name=name))
        with (
            mock.patch.object(base_framework.importlib, "import_module") as importer,
            mock.patch.object(base_framework, "_auto_import_framework_modules") as discover,
            mock.patch.dict(base_framework.FRAMEWORK_REGISTRY._registry, {name: constructor}, clear=True),
        ):
            result = base_framework.build_framework(cfg)
        constructor.assert_called_once_with(cfg)
        self.assertIs(result, constructor.return_value)
        return importer, discover

    def test_ego4wam_imports_only_its_framework(self):
        importer, discover = self._build("Ego4WAM")
        importer.assert_called_once_with("starVLA.model.framework.WM4A.Ego4WAM")
        discover.assert_not_called()

    def test_other_frameworks_keep_existing_discovery(self):
        importer, discover = self._build("OtherFramework")
        importer.assert_not_called()
        discover.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
