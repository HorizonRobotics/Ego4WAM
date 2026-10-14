# Copyright (c) 2026 Beijing Horizon Information Technology Company Limited.
# Author: Ego4WAM Team.
# SPDX-License-Identifier: Apache-2.0
"""Check Ego4WAM notices without changing inherited licenses or file formats."""

import ast
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
    "LICENSES/DiffusionPolicy-MIT.txt",
    "LICENSES/OpenVLA-MIT.txt",
    "LICENSES/OpenPI-Apache-2.0.txt",
    "LICENSES/MsgpackNumpy-BSD-3-Clause.txt",
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


def _provenance_only_update(name):
    """Source-only notices must not claim authorship of unchanged implementations."""
    path = ROOT / name
    if name.endswith(".license"):
        path = path.with_suffix("")
    original = _git("show", f"{UPSTREAM}:{path.relative_to(ROOT)}")
    if original.returncode:
        return False
    current = path.read_text()
    if current == original.stdout:
        return True
    if path.suffix == ".py":
        return ast.dump(ast.parse(current)) == ast.dump(ast.parse(original.stdout))
    if path.name == "Makefile":
        def executable_lines(text):
            return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]

        return executable_lines(current) == executable_lines(original.stdout)
    return False


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
        if OWNER not in header and _provenance_only_update(name):
            assert "Source provenance: StarVLA project" in header, name
            assert "SPDX-License-Identifier:" in header, name
            assert "Author: Ego4WAM Team." not in header, name
            continue
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


def _tracked_files():
    files = _git("ls-files")
    if files.returncode:
        pytest.skip("Git source inventory is unavailable")
    return files.stdout.splitlines()


def _header(path):
    source = path.read_text()
    if path.suffix == ".py":
        tree = ast.parse(source)
        if tree.body:
            return "\n".join(source.splitlines()[:tree.body[0].lineno - 1])
    return "\n".join(source.splitlines()[:20])


def test_entire_release_has_license_and_copyright_or_source_provenance():
    for name in _tracked_files():
        if name in LICENSE_TEXTS or name == "NOTICE":
            continue
        path = ROOT / name
        companion = Path(str(path) + ".license")
        header = _header(companion if companion.exists() else path)
        assert "SPDX-License-Identifier:" in header, name
        has_owner = re.search(r"Copyright (?:\(c\)|\d{4})", header)
        assert has_owner or "Source provenance:" in header, name


def test_starvla_file_copyright_is_not_inferred_from_project_origin():
    counterparts = {
        "deployment/model_server/tools/ego4wam_websocket_policy_server.py":
            "deployment/model_server/tools/websocket_policy_server.py",
    }
    if _git("cat-file", "-e", UPSTREAM).returncode:
        pytest.skip("Upstream history is unavailable")
    for name in _tracked_files():
        if name in LICENSE_TEXTS or name == "NOTICE":
            continue
        path = ROOT / name
        header = _header(path)
        for line in header.splitlines():
            if not re.search(r"Copyright.*starvla", line, re.IGNORECASE):
                continue
            source = _git("show", f"{UPSTREAM}:{counterparts.get(name, name)}")
            assert source.returncode == 0, name
            declaration = " ".join(line.removeprefix("# ").split())
            assert any(
                " ".join(candidate.removeprefix("# ").split()) == declaration
                for candidate in source.stdout.splitlines()[:30]
            ), (name, line)


def test_original_third_party_license_notices_are_archived():
    assert "2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti" in (ROOT / "LICENSES/OpenVLA-MIT.txt").read_text()
    assert "2013-2022, Lev E. Givon" in (ROOT / "LICENSES/MsgpackNumpy-BSD-3-Clause.txt").read_text()
    assert "Apache License" in (ROOT / "LICENSES/OpenPI-Apache-2.0.txt").read_text()
    assert '"LICENSES/*.txt"' in (ROOT / "pyproject.toml").read_text()
