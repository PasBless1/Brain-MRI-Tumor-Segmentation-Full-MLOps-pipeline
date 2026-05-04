"""Model monitoring: data drift detection and performance tracking with Evidently."""
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
import mlflow
import logging
import os

logger = logging.getLogger(__name__)


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                     report_dir: str = "reports/monitoring/") -> None:
    os.makedirs(report_dir, exist_ok=True)
    column_mapping = ColumnMapping(target="has_mask", prediction="prediction")

    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    report_path = os.path.join(report_dir, "drift_report.html")
    report.save_html(report_path)
    mlflow.log_artifact(report_path)
    logger.info(f"Drift report saved to {report_path}")


def check_performance_degradation(current_dice: float, baseline_dice: float,
                                   threshold: float = 0.05) -> bool:
    degradation = baseline_dice - current_dice
    if degradation > threshold:
        logger.warning(f"Model degradation detected: baseline={baseline_dice:.4f}, current={current_dice:.4f}")
        return True
    logger.info(f"Model performance OK: baseline={baseline_dice:.4f}, current={current_dice:.4f}")
    return False
