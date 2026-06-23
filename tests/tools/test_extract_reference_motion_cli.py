from __future__ import annotations

from pathlib import Path

from tools.avatar_assets import extract_reference_motion as erm


def test_extract_reference_motion_cli_uses_one_input_and_two_outputs(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "reference.mp4"
    input_path.write_bytes(b"fake")

    called = {}

    def _fake_extract(input_path_arg: Path, output_dir: Path):
        called["input"] = input_path_arg
        called["output"] = output_dir
        return {
            "hand_motion": {"path": str(output_dir / "hand_motion.npz"), "summary": {}},
            "body_motion": {"path": str(output_dir / "body_and_hands_motion.npz"), "summary": {}},
        }

    monkeypatch.setattr(erm, "extract_reference_motion", _fake_extract)

    exit_code = erm.main(["--input", str(input_path), "--output-dir", str(tmp_path / "motion")])

    assert exit_code == 0
    assert called["input"] == input_path
    assert called["output"] == tmp_path / "motion"
