# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0
"""Check Ego4WAM notices without changing inherited licenses or file formats."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "4507931a625c844404c4a76128fb116536d8ca7c"
OWNER = "Copyright (c) 2026 Beijing Horizon Information Technology Company Limited."
AUTHOR = "Author: Ego4WAM Team."
EVAL = "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/"
# License texts describe the applicable terms, not ownership of the license prose.
LICENSE_TEXTS = {
    "LICENSE",
    "LICENSES/FastWAM-MIT.txt",
    "LICENSES/RoboDojo-MIT.txt",
    "LICENSES/StarVLA-MIT.txt",
    "LICENSES/XPolicyLab-Apache-2.0.txt",
    EVAL + "patches/RoboDojo-LICENSE",
    EVAL + "patches/XPolicyLab-LICENSE",
    "starVLA/training/trainer_utils/assets/Apache-2.0.txt",
}


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _ego4wam_changes():
    if _git("cat-file", "-e", UPSTREAM).returncode:
        pytest.skip("Upstream Git history is not available in this source distribution")
    changes = _git("diff", "--name-only", "--diff-filter=AM", UPSTREAM, "--")
    assert changes.returncode == 0, changes.stderr
    paths = changes.stdout.splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard")
    assert untracked.returncode == 0, untracked.stderr
    return sorted(set(paths + untracked.stdout.splitlines()))


def test_all_ego4wam_changes_have_scoped_company_and_author_notices():
    checked = []
    for name in _ego4wam_changes():
        if name in LICENSE_TEXTS:
            continue
        path = ROOT / name
        notice = Path(str(path) + ".license")
        if not notice.exists():
            notice = path
        header = "\n".join(notice.read_text().splitlines()[:16])
        assert OWNER in header, name
        assert AUTHOR in header, name
        if name != "NOTICE":
            assert "SPDX-License-Identifier: Apache-2.0" in header, name
        checked.append(name)
    assert len(checked) >= 80


def test_modified_upstream_files_retain_original_header_notices():
    for name in _ego4wam_changes():
        if name in LICENSE_TEXTS or name.endswith(".license"):
            continue
        original = _git("show", f"{UPSTREAM}:{name}")
        if original.returncode:
            continue  # Added files have no counterpart at the upstream revision.
        current = (ROOT / name).read_text()
        for line in original.stdout.splitlines()[:30]:
            if re.search(r"copyright|licensed under|implemented by", line, re.IGNORECASE):
                assert line in current, (name, line)


def test_machine_readable_notices_have_explicit_modification_scope():
    paths = list((ROOT / "starVLA/dataloader/ego4wam/assets").glob("*.license"))
    paths += list((ROOT / EVAL).rglob("*.license"))
    paths += list((ROOT / "examples/simBenchmarks/RoboDojo/train_files_ego4wam").glob("*.license"))
    assert len(paths) == 9
    for path in paths:
        assert path.with_suffix("").is_file(), path
        assert "Ego4WAM additions/modifications and packaging only" in path.read_text(), path
    metadata = (ROOT / "pyproject.toml").read_text()
    assert '"assets/*.license"' in metadata


def test_original_starvla_license_is_preserved_verbatim():
    original = _git("show", f"{UPSTREAM}:LICENSE")
    if original.returncode:
        pytest.skip("Upstream Git history is not available in this source distribution")
    assert (ROOT / "LICENSES/StarVLA-MIT.txt").read_text() == original.stdout
