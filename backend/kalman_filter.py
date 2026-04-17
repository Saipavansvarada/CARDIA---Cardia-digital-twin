"""
CARDIA — Kalman Filter
Smooths noisy HR + HRV readings from MAX30102 sensor.
All ML models use this cleaned data for better accuracy.
"""

import numpy as np
from config import KALMAN


class CardiacKalmanFilter:
    """
    1D Kalman Filter tuned for heart rate smoothing.

    Simple explanation:
    - Every HR reading from MAX30102 has noise (+-5 BPM)
    - Kalman blends new reading with predicted value
    - Result: smooth accurate HR that reacts to real changes
      but ignores random sensor jumps
    """

    def __init__(self, initial_value: float = 70.0,
                 process_noise: float = None,
                 measurement_noise: float = None):
        self.x = initial_value
        self.P = 10.0
        self.Q = process_noise     or KALMAN["hr_process_noise"]
        self.R = measurement_noise or KALMAN["hr_measurement_noise"]
        self.raw_history      = []
        self.filtered_history = []

    def update(self, raw_value: float) -> float:
        """Takes raw reading, returns smoothed value."""
        x_pred = self.x
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (raw_value - x_pred)
        self.P = (1 - K) * P_pred
        self.raw_history.append(round(raw_value, 1))
        self.filtered_history.append(round(self.x, 1))
        return round(self.x, 1)

    def reset(self, initial_value: float = 70.0):
        """Reset filter for new session."""
        self.x = initial_value
        self.P = 10.0
        self.raw_history      = []
        self.filtered_history = []

    def get_noise_reduction(self) -> float:
        """Returns noise reduction percentage."""
        if len(self.raw_history) < 2:
            return 0.0
        raw_std      = float(np.std(self.raw_history))
        filtered_std = float(np.std(self.filtered_history))
        if raw_std == 0:
            return 0.0
        return round((1 - filtered_std / raw_std) * 100, 1)


class HRVKalmanFilter:
    """Kalman Filter tuned for HRV smoothing."""

    def __init__(self, initial_value: float = 45.0):
        self.x = initial_value
        self.P = 15.0
        self.Q = KALMAN["hrv_process_noise"]
        self.R = KALMAN["hrv_measurement_noise"]
        self.raw_history      = []
        self.filtered_history = []

    def update(self, raw_value: float) -> float:
        x_pred = self.x
        P_pred = self.P + self.Q
        K      = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (raw_value - x_pred)
        self.P = (1 - K) * P_pred
        self.raw_history.append(round(raw_value, 1))
        self.filtered_history.append(round(self.x, 1))
        return round(self.x, 1)

    def reset(self, initial_value: float = 45.0):
        self.x = initial_value
        self.P = 15.0
        self.raw_history      = []
        self.filtered_history = []

    def get_noise_reduction(self) -> float:
        if len(self.raw_history) < 2:
            return 0.0
        raw_std      = float(np.std(self.raw_history))
        filtered_std = float(np.std(self.filtered_history))
        if raw_std == 0:
            return 0.0
        return round((1 - filtered_std / raw_std) * 100, 1)


class CardiacDataSmoother:
    """
    Combined smoother for all cardiac signals.
    This is what the main app uses.

    Usage:
        smoother = CardiacDataSmoother()
        result = smoother.process(raw_hr=145, raw_hrv=32)
        print(result["clean_hr"], result["clean_hrv"])
    """

    def __init__(self):
        self.hr_filter  = CardiacKalmanFilter(
            initial_value=70.0,
            process_noise=KALMAN["hr_process_noise"],
            measurement_noise=KALMAN["hr_measurement_noise"]
        )
        self.hrv_filter = HRVKalmanFilter(initial_value=45.0)

    def process(self, raw_hr: float, raw_hrv: float) -> dict:
        """Process one reading and return cleaned values."""
        clean_hr  = self.hr_filter.update(raw_hr)
        clean_hrv = self.hrv_filter.update(raw_hrv)

        hr_change  = 0.0
        hrv_change = 0.0
        if len(self.hr_filter.filtered_history) > 1:
            hr_change  = clean_hr  - self.hr_filter.filtered_history[-2]
            hrv_change = clean_hrv - self.hrv_filter.filtered_history[-2]

        return {
            "raw_hr":     raw_hr,
            "raw_hrv":    raw_hrv,
            "clean_hr":   clean_hr,
            "clean_hrv":  clean_hrv,
            "hr_change":  round(hr_change,  1),
            "hrv_change": round(hrv_change, 1),
        }

    def reset(self, initial_hr: float = 70.0,
              initial_hrv: float = 45.0):
        """Reset both filters for new session."""
        self.hr_filter.reset(initial_hr)
        self.hrv_filter.reset(initial_hrv)

    def get_stats(self) -> dict:
        """Returns noise reduction stats."""
        return {
            "hr_noise_reduced":  self.hr_filter.get_noise_reduction(),
            "hrv_noise_reduced": self.hrv_filter.get_noise_reduction(),
            "total_readings":    len(self.hr_filter.filtered_history),
        }


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("CARDIA — Kalman Filter Test")
    print("=" * 50)
    print()

    smoother = CardiacDataSmoother()

    print(f"{'#':<6} {'Raw HR':<12} {'Clean HR':<12} "
          f"{'Raw HRV':<12} {'Clean HRV'}")
    print("-" * 55)

    true_hrs = list(range(70, 160, 5))
    for i, true_hr in enumerate(true_hrs):
        noisy_hr  = true_hr + np.random.normal(0, 5)
        noisy_hrv = max(10, 80 - true_hr * 0.4 + np.random.normal(0, 8))
        result    = smoother.process(noisy_hr, noisy_hrv)
        print(f"{i+1:<6} {result['raw_hr']:<12.1f} "
              f"{result['clean_hr']:<12.1f} "
              f"{result['raw_hrv']:<12.1f} "
              f"{result['clean_hrv']:.1f}")

    stats = smoother.get_stats()
    print()
    print(f"HR  noise reduced:  {stats['hr_noise_reduced']}%")
    print(f"HRV noise reduced:  {stats['hrv_noise_reduced']}%")
    print(f"Total readings:     {stats['total_readings']}")
    print()
    print("Kalman Filter ready!")