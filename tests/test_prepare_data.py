import os
import pandas as pd
import pytest

os.environ["TEST_MODE"] = "1"


@pytest.fixture
def sample_raw_data(tmp_path, monkeypatch):
    """Create small synthetic dataset for testing."""
    df = pd.DataFrame({
        "R1": [5, 4, 3],
        "R2": [5, 4, 3],
        "I1": [4, 3, 5],
        "I2": [3, 2, 5],
        "A1": [2, 3, 4],
        "A2": [2, 3, 4],
        "S1": [1, 5, 3],
        "S2": [1, 5, 3],
        "E1": [4, 4, 4],
        "E2": [4, 4, 4],
        "C1": [3, 3, 3],
        "C2": [3, 3, 3],
        "major": ["psychology", "biology", "biology"],
    })

    d = tmp_path / "data"
    d.mkdir()
    file = d / "data.csv"
    df.to_csv(file, sep="\t", index=False)

    # Patch working directory so prepare_data writes into tmp path
    monkeypatch.chdir(tmp_path)
    return file


def test_prepare_data_runs_without_errors(sample_raw_data):
    """Ensure the function runs and produces output."""
    from prepare_data import run_prepare_data

    output_file = "data/final_data.csv"

    df_final = run_prepare_data(
        input_path="data/data.csv",
        output_path=output_file,
        test_mode=True
    )

    assert os.path.exists(output_file), "Output file was not created"
    assert len(df_final) > 0, "Output dataframe is empty"


def test_prepare_data_output_columns(sample_raw_data):
    """Verify expected columns exist."""
    from prepare_data import run_prepare_data

    run_prepare_data(
        input_path="data/data.csv",
        output_path="data/final_data.csv",
        test_mode=True
    )

    df_final = pd.read_csv("data/final_data.csv")

    required = [
        "R_pct", "I_pct", "A_pct",
        "S_pct", "E_pct", "C_pct",
        "major_standard",
    ]

    for col in required:
        assert col in df_final.columns, f"Missing column: {col}"
