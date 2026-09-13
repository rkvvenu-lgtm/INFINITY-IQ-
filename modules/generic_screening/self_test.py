"""
SIH26170 - Generic Screening: Algorithm Self-Test
==================================================
Synthetic correctness validation for the generic screening algorithm.

Because arbitrary datasets have no ground-truth specification, the
generic algorithm is validated with controlled synthetic data:

TEST 1 - Clean data
    Three parameters drawn from a standard normal distribution.
    Expectation: nearly zero anomalies (low false-positive rate).

TEST 2 - Injected outliers
    Same clean population PLUS 10 injected records whose P1 value is
    6 sigma above the mean.
    Expectation: at least 90% of injected records are flagged.

Result
------
    status  : PASS / FAIL
    metrics : per-test numbers and the thresholds used
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from modules.generic_screening.generic_engine import GenericScreeningEngine


class GenericSelfTest:
    """Documented, reproducible synthetic validation."""

    VERSION = "1.0"

    CLEAN_ROWS = 300
    OUTLIER_ROWS = 10
    OUTLIER_SIGMA = 6.0
    RECALL_THRESHOLD = 0.90
    FALSE_POSITIVE_THRESHOLD = 0.10
    SEED = 42

    def __init__(
        self,
        engine: Any = None,
    ) -> None:
        self.engine = engine or GenericScreeningEngine()

    def run(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.SEED)

        clean = pd.DataFrame(
            {
                "P1": rng.normal(0, 1, self.CLEAN_ROWS),
                "P2": rng.normal(0, 1, self.CLEAN_ROWS),
                "P3": rng.normal(0, 1, self.CLEAN_ROWS),
            }
        )

        # ----------------------------------------------------------
        # TEST 1 - clean data
        # ----------------------------------------------------------
        clean_screen = self.engine.screen(clean, mode="LIVE_FUTURE_SCREENING")
        clean_result = clean_screen["result"]
        clean_flagged = int(
            clean_result["Anomaly_Flag"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        false_positive_rate = clean_flagged / self.CLEAN_ROWS

        # ----------------------------------------------------------
        # TEST 2 - injected outliers
        # ----------------------------------------------------------
        outliers = pd.DataFrame(
            {
                "P1": np.full(
                    self.OUTLIER_ROWS,
                    self.OUTLIER_SIGMA,
                ),
                "P2": rng.normal(0, 1, self.OUTLIER_ROWS),
                "P3": rng.normal(0, 1, self.OUTLIER_ROWS),
            }
        )
        injected = pd.concat(
            [clean, outliers],
            ignore_index=True,
        )

        injected_screen = self.engine.screen(
            injected,
            mode="LIVE_FUTURE_SCREENING",
        )
        injected_result = injected_screen["result"]

        injected_flags = (
            injected_result["Anomaly_Flag"]
            .fillna(False)
            .astype(bool)
            .iloc[self.CLEAN_ROWS:]
        )
        detected = int(injected_flags.sum())
        recall = detected / self.OUTLIER_ROWS

        status = "PASS"
        failures = []
        if recall < self.RECALL_THRESHOLD:
            status = "FAIL"
            failures.append(
                f"outlier recall {recall:.2f} < "
                f"{self.RECALL_THRESHOLD:.2f}"
            )
        if false_positive_rate > self.FALSE_POSITIVE_THRESHOLD:
            status = "FAIL"
            failures.append(
                f"clean false-positive rate {false_positive_rate:.2f} > "
                f"{self.FALSE_POSITIVE_THRESHOLD:.2f}"
            )

        return {
            "status": status,
            "failures": failures,
            "version": self.VERSION,
            "seed": self.SEED,
            "test_1_clean": {
                "rows": self.CLEAN_ROWS,
                "flagged": clean_flagged,
                "false_positive_rate": round(
                    false_positive_rate,
                    4,
                ),
                "threshold": self.FALSE_POSITIVE_THRESHOLD,
            },
            "test_2_outliers": {
                "rows": self.OUTLIER_ROWS,
                "injected_sigma": self.OUTLIER_SIGMA,
                "detected": detected,
                "recall": round(recall, 4),
                "threshold": self.RECALL_THRESHOLD,
            },
            "method": (
                "robust z-score (median/MAD) + Isolation Forest; "
                "a row is flagged when blended Anomaly_Risk >= 0.5 "
                "or max robust z >= 3.5 sigma."
            ),
        }


def run_self_test() -> Dict[str, Any]:
    """Convenience wrapper."""
    return GenericSelfTest().run()