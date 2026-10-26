"""stage-info.json is what lets a comparison prove it used one GPU: an allow-list of environment names, never a secret."""
import json

from tensorpilot import stage_info


def test_only_allow_listed_environment_names_are_copied(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-9")
    monkeypatch.setenv("TENSORPILOT_API_KEY", "super-secret-key")
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 30)
    monkeypatch.setenv("SOMETHING_ELSE", "x")
    info = stage_info.collect("7b-bf16-prod")
    assert info["environment"] == {"RUNPOD_POD_ID": "pod-9"}
    assert "super-secret-key" not in json.dumps(info) and "hf_" not in json.dumps(info)


def test_gpus_are_parsed_from_nvidia_smi_and_absent_without_it(monkeypatch):
    class Done:
        stdout = "NVIDIA GeForce RTX 4090, GPU-abc, 595.91.07, 24564\n"
    monkeypatch.setattr(stage_info.subprocess, "run", lambda *a, **k: Done())
    assert stage_info.gpus() == [{"name": "NVIDIA GeForce RTX 4090", "uuid": "GPU-abc", "driver_version": "595.91.07",
                                  "memory_total_mib": "24564"}]

    def missing(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(stage_info.subprocess, "run", missing)
    assert stage_info.gpus() == []


def test_build_info_is_read_when_present_and_none_when_not(tmp_path, monkeypatch):
    path = tmp_path / "BUILD_INFO.json"
    monkeypatch.setattr(stage_info, "BUILD_INFO", path)
    assert stage_info.build_info() is None
    path.write_text('{"source_sha": "abc", "image_tag": "v1"}')
    assert stage_info.build_info() == {"source_sha": "abc", "image_tag": "v1"}


def test_the_command_line_writes_the_file(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["stage_info", "--stage", "x", "--output", str(tmp_path / "i.json")])
    monkeypatch.setattr(stage_info.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert stage_info.main() == 0
    assert json.loads((tmp_path / "i.json").read_text())["schema"] == stage_info.SCHEMA
