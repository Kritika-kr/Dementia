"""
Appends the current run's results to a persistent experiment log, so every
time you change something (add a feature, tune differently) and re-run
generate_paper_assets.py, you keep a full history instead of overwriting it.

Usage: import and call after generate_paper_assets.py produces its tables:
    from src.experiment_log import log_run
    log_run("run4_my_change", model_comparison_df, ablation_df)

Or run standalone to re-log whatever is currently in outputs/tables/
model_comparison.csv and ablation_results.csv under a given run name:
    python -m src.experiment_log run4_my_change
"""
import sys
import pandas as pd
from . import config

MODELS_LOG = config.OUTPUT_DIR / "tables" / "experiment_log_models.csv"
ABLATION_LOG = config.OUTPUT_DIR / "tables" / "experiment_log_ablation.csv"


def log_run(run_name: str, model_comparison_df: pd.DataFrame, ablation_df: pd.DataFrame):
    models = model_comparison_df.reset_index().rename(columns={"index": "model"})
    models.insert(0, "run", run_name)

    ablation = ablation_df.reset_index()
    ablation.insert(0, "run", run_name)

    for df, path in [(models, MODELS_LOG), (ablation, ABLATION_LOG)]:
        if path.exists():
            existing = pd.read_csv(path)
            existing = existing[existing["run"] != run_name]  # replace if re-run with same name
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(path, index=False)

    print(f"Logged run '{run_name}' to {MODELS_LOG} and {ABLATION_LOG}")


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "unnamed_run"
    models_path = config.OUTPUT_DIR / "tables" / "model_comparison.csv"
    ablation_path = config.OUTPUT_DIR / "tables" / "ablation_results.csv"
    models_df = pd.read_csv(models_path, index_col=0)
    ablation_df = pd.read_csv(ablation_path, index_col=0)
    log_run(run_name, models_df, ablation_df)