# Copyright (c) 2026 Ego4WAM Authors.
# SPDX-License-Identifier: Apache-2.0
"""Release documentation links, command syntax and license-scope guards."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_root_license_and_package_metadata_are_apache():
    assert (ROOT / "LICENSE").read_bytes() == (
        ROOT / "starVLA/training/trainer_utils/assets/Apache-2.0.txt"
    ).read_bytes()
    metadata = (ROOT / "pyproject.toml").read_text()
    assert 'license = "Apache-2.0"' in metadata
    assert '"setuptools>=77"' in metadata
    assert '"License :: OSI Approved :: MIT License"' not in metadata
    for item in ('"NOTICE"', '"LICENSES/*.txt"', '"starVLA/model/framework/VM4A/_dp_vendor/LICENSE"'):
        assert item in metadata
    assert "Ego4WAM Authors" in (ROOT / "NOTICE").read_text()


def test_upstream_license_copies_and_headers_are_retained():
    original = (ROOT / "LICENSES/StarVLA-MIT.txt").read_text()
    assert "MIT License" in original
    assert "Copyright (c) StarVLA Team." in original
    assert "Preserve attribution" in original
    assert "Copyright (c) 2026 The FastWAM Authors." in (ROOT / "LICENSES/FastWAM-MIT.txt").read_text()
    patches = ROOT / "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/patches"
    for name, source in (("RoboDojo-MIT.txt", "RoboDojo-LICENSE"), ("XPolicyLab-Apache-2.0.txt", "XPolicyLab-LICENSE")):
        assert (ROOT / "LICENSES" / name).read_bytes() == (patches / source).read_bytes()
    assert "MIT License" in (ROOT / "starVLA/model/framework/VM4A/_dp_vendor/LICENSE").read_text()
    assert "Licensed under the MIT License" in (ROOT / "starVLA/model/modules/world_model/Wan2.py").read_text()
    for file in ("starVLA/model/framework/WM4A/Ego4WAM.py", "starVLA/model/modules/world_model/WanMoT.py"):
        header = (ROOT / file).read_text().splitlines()[:8]
        assert "The FastWAM Authors." in header[0]
        assert "# SPDX-License-Identifier: Apache-2.0 AND MIT" in header


def test_ego4wam_authored_headers_include_apache():
    checked = []
    for tree in ("starVLA", "deployment", "examples/simBenchmarks/RoboDojo", "tests"):
        for path in (ROOT / tree).rglob("*"):
            if path.suffix not in {".py", ".sh", ".yaml"}:
                continue
            header = "\n".join(path.read_text().splitlines()[:12])
            if "Copyright (c) 2026 Ego4WAM Authors." not in header:
                continue
            assert "SPDX-License-Identifier: Apache-2.0" in header, path
            assert "SPDX-License-Identifier: MIT\n" not in header, path
            checked.append(path)
    assert len(checked) >= 50


@pytest.mark.parametrize(
    "filename",
    [
        "README.md",
        "docs/RoboDojo.md",
        "LICENSES/README.md",
        "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/README.md",
    ],
)
def test_local_document_links_resolve(filename):
    path = ROOT / filename
    for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        relative = target.split("#", 1)[0]
        assert (path.parent / relative).exists(), (filename, target)


@pytest.mark.parametrize("filename", ["README.md", "docs/RoboDojo.md"])
def test_documented_bash_blocks_parse_without_executing(filename):
    blocks = re.findall(r"```bash\n(.*?)```", (ROOT / filename).read_text(), re.DOTALL)
    assert blocks
    for block in blocks:
        result = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_readme_points_to_ego4wam_recipes_and_historical_evaluation():
    text = (ROOT / "README.md").read_text()
    assert "## 🔥 RoboDojo Post-training" in text
    assert "## 🤖 RoboDojo Evaluation" in text
    assert "🏋" not in text
    assert "RoboDojo Posttraining" not in text
    assert "historical RoboDojo version used in the paper" in text
    for filename in ("README.md", "docs/RoboDojo.md", "examples/simBenchmarks/RoboDojo/eval_files_ego4wam/README.md"):
        headings = [line for line in (ROOT / filename).read_text().splitlines() if line.startswith("#")]
        assert all("historical" not in line.lower() for line in headings)
    assert "License-Apache_2.0" in text
    assert "License-MIT" not in text
    for source in ("wan", "video_pretrained"):
        for mode in ("base", "joint"):
            name = f"train_ego4wam_{source}_robodojo_{mode}.sh"
            assert name in text
            assert (ROOT / "examples/simBenchmarks/RoboDojo/train_files_ego4wam" / name).is_file()
    assert "eval_files_ego4wam/local" not in text or "local_server.sh" in text
    for name in ("local_server.sh", "local_client.sh", "summarize_results.py", "NUM_EPISODES=10", "NUM_EPISODES=native"):
        assert name in text
    assert "not the current official leaderboard/submission protocol" in text
    assert "not yet published on HF" in text
    assert "RoboDojo/train_files/" not in text
