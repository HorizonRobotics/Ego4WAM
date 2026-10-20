# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: MIT
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
            mock.patch.dict(base_framework.FRAMEWORK_REGISTRY._registry, {name: constructor}, clear=True),
        ):
            result = base_framework.build_framework(cfg)
        constructor.assert_called_once_with(cfg)
        self.assertIs(result, constructor.return_value)
        return importer

    def test_ego4wam_imports_only_its_framework(self):
        importer = self._build("Ego4WAM")
        importer.assert_called_once_with("starVLA.model.framework.WM4A.Ego4WAM")

    def test_unreleased_frameworks_fail_before_any_import(self):
        cfg = SimpleNamespace(framework=SimpleNamespace(name="OtherFramework"))
        with mock.patch.object(base_framework.importlib, "import_module") as importer:
            with self.assertRaisesRegex(NotImplementedError, "only supports Ego4WAM"):
                base_framework.build_framework(cfg)
            importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
