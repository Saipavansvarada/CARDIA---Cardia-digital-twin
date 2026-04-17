# backend/shap_explain.py
import shap
import xgboost as xgb
import numpy as np

class ShapExplainer:
    def __init__(self, model_path="xgboost_ces_model.json"):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, features):
        shap_values = self.explainer.shap_values(np.array([features]))
        importance = dict(zip(range(len(features)), shap_values[0]))
        # Return top 3 factors
        top3 = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        return top3