"""
CARDIA — RL Agent (Strategy Recommender)
Picks the best training strategy based on current cardiac state.
Uses rule-based logic for hackathon (stable + fast).
"""

from config import STRATEGIES, SAFETY, USER

MAX_HR     = USER["max_hr"]
RESTING_HR = USER["resting_hr"]

STRATEGY_NAMES = {
    0: "HIIT",
    1: "HRV Breathing",
    2: "Active Recovery",
    3: "LISS Cardio",
    4: "Full Rest",
}


class CardiacRLAgent:
    """
    Smart rule-based RL agent for strategy selection.
    Maps cardiac state → best training strategy.

    State:  [ces, hrv, load, sleep, spo2, safety]
    Action: 0=HIIT, 1=HRV Breathing, 2=Recovery, 3=LISS, 4=Full Rest
    """

    def predict(self, state):
        """
        Args:
            state: list or array [ces, hrv, load, sleep, spo2, safety_code]
                   safety_code: 0=SAFE, 1=CAUTION, 2=DANGER
        Returns:
            (action, None) — action is 0-4
        """
        ces    = float(state[0]) if len(state) > 0 else 60
        hrv    = float(state[1]) if len(state) > 1 else 40
        load   = float(state[2]) if len(state) > 2 else 3
        sleep  = float(state[3]) if len(state) > 3 else 7
        spo2   = float(state[4]) if len(state) > 4 else 98
        safety = int(state[5])   if len(state) > 5 else 0

        # ── Priority rules ────────────────────────────────────────────────────
        # DANGER → Full Rest always
        if safety == 2:
            return 4, None

        # Low SpO2 → Full Rest
        if spo2 < SAFETY["caution_spo2"]:
            return 4, None

        # CAUTION → Active Recovery
        if safety == 1:
            return 2, None

        # Very low CES → Active Recovery
        if ces < 35:
            return 2, None

        # Low HRV → HRV Breathing Protocol
        if hrv < 28:
            return 1, None

        # High load + low sleep → LISS (low intensity)
        if load > 7 and sleep < 6:
            return 3, None

        # Low CES + ok HRV → LISS
        if ces < 55:
            return 3, None

        # Good state → HIIT
        if ces >= 70 and hrv >= 35 and sleep >= 7:
            return 0, None

        # Default → LISS
        return 3, None

    def get_strategy_detail(self, action):
        """Returns full strategy details."""
        return STRATEGIES.get(int(action), STRATEGIES[3])


def load_agent():
    """Load and return the RL agent. Used by main.py."""
    return CardiacRLAgent()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("CARDIA — RL Agent Test")
    print("=" * 50)
    print()

    agent = load_agent()

    test_cases = [
        {"label": "Perfect athlete",    "state": [85, 55, 3, 8, 99, 0]},
        {"label": "Moderate fitness",   "state": [60, 35, 5, 7, 98, 0]},
        {"label": "Low HRV",            "state": [55, 22, 4, 6, 97, 0]},
        {"label": "Caution zone",       "state": [45, 25, 8, 5, 95, 1]},
        {"label": "Danger zone",        "state": [30, 15, 9, 4, 91, 2]},
        {"label": "Tired + overloaded", "state": [50, 30, 8, 5, 97, 0]},
    ]

    print(f"{'State':<25} {'Action':<6} {'Strategy'}")
    print("-" * 55)
    for t in test_cases:
        action, _ = agent.predict(t["state"])
        name      = STRATEGY_NAMES[action]
        print(f"  {t['label']:<23} {action:<6} {name}")

    print()
    print("RL Agent ready!")