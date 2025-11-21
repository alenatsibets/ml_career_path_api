import os
import pandas as pd
import pytest

from prepare_data_48 import run_prepare_data_48


@pytest.fixture
def sample_raw_data_48(tmp_path, monkeypatch):
    data = {f"{c}{i}": [5, 4, 3] for c in "RIASEC" for i in range(1, 9)}
    data["major"] = ["psychology", "biology", "biology"]

    df = pd.DataFrame(data)

    d = tmp_path / "data"
    d.mkdir()
    raw_file = d / "data.csv"
    df.to_csv(raw_file, sep="\t", index=False)

    monkeypatch.chdir(tmp_path)
    return raw_file


def test_prepare_data_48_runs(sample_raw_data_48):
    df_out = run_prepare_data_48(
        input_path="data/data.csv",
        output_path="data/final_data_48.csv",
        test_mode=True
    )

    assert os.path.exists("data/final_data_48.csv")
    assert len(df_out) > 0


def test_prepare_data_48_columns(sample_raw_data_48):
    df_out = run_prepare_data_48(
        input_path="data/data.csv",
        output_path="data/final_data_48.csv",
        test_mode=True
    )

    expected_count = 48 + 2  # 48 items + 1 major + 1 major_standard
    assert df_out.shape[1] == expected_count, f"Expected {expected_count} columns"
