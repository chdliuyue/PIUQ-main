from pathlib import Path

from piuq.config import Config, load_config


def test_env_override(tmp_path, monkeypatch):
    cfg_path = tmp_path / "base.yaml"
    cfg_path.write_text("preprocess:\n  history_sec: 3.0\n")
    monkeypatch.setenv("PREPROCESS_HISTORY_SEC", "5.0")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.preprocess.history_sec == 5.0


def test_override_file(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("preprocess:\n  history_sec: 3.0\n")
    override = tmp_path / "override.yaml"
    override.write_text("preprocess:\n  history_sec: 4.0\n")
    cfg = load_config(base, [override])
    assert cfg.preprocess.history_sec == 4.0
