import os
from util.logger import get_logger


def test_logger_creates_log_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    logger = get_logger("test", "test.log")
    logger.info("Hello")

    assert os.path.exists("logs/test.log")
