import pandas as pd
import os
import logging
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import DriftedColumnsCount

logger = logging.getLogger("monitoring")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


TRAIN_DATA_PATH = "data/creditcard.csv"
PREDICTION_LOG_PATH = "reports/prediction_log.csv"
DRIFT_REPORT_PATH = "reports/drift_report.html"

DRIFT_THRESHOLD = 0.5


def load_reference_data() -> pd.DataFrame:
    """Load training data as reference dataset."""
    try:
        df = pd.read_csv(TRAIN_DATA_PATH)
        logger.debug("Reference data loaded: %s", df.shape)
        return df

    except FileNotFoundError:
        logger.error(
            "Training data not found at %s",
            TRAIN_DATA_PATH
        )
        raise

    except Exception:
        logger.exception(
            "Unexpected error loading reference data"
        )
        raise


def load_prediction_log() -> pd.DataFrame:
    """Load logged predictions as current dataset."""
    try:
        df = pd.read_csv(PREDICTION_LOG_PATH)
        logger.debug("Prediction log loaded: %s", df.shape)
        return df

    except FileNotFoundError:
        logger.error(
            "Prediction log not found at %s",
            PREDICTION_LOG_PATH
        )
        raise

    except Exception:
        logger.exception(
            "Unexpected error loading prediction log"
        )
        raise


def run_drift_detection():
    """Compare prediction data against training data."""

    try:
        reference = load_reference_data()
        current = load_prediction_log()

        common_cols = ["Amount", "Time"]

        reference_subset = reference[common_cols]
        current_subset = current[common_cols]

        if len(current_subset) < 10:
            logger.warning(
                "Not enough prediction logs for drift detection. "
                "Need at least 10 predictions."
            )
            return

        report = Report(
            metrics=[
                DataDriftPreset(),
                DriftedColumnsCount()
            ]
        )

        snapshot = report.run(
            reference_data=reference_subset,
            current_data=current_subset
        )

        os.makedirs("reports", exist_ok=True)

        snapshot.save_html(DRIFT_REPORT_PATH)

        logger.info(
            "Drift report saved to %s",
            DRIFT_REPORT_PATH
        )

        return snapshot

    except Exception:
        logger.exception(
            "Error during drift detection"
        )
        raise


if __name__ == "__main__":
    run_drift_detection()