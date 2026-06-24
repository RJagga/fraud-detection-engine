import shap
import joblib
import numpy as np
import pandas as pd

model = joblib.load('artifacts/xgboost_model.pkl')

# TreeExplainer is fast for tree-based models — exact Shapley values, not approximate
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, 'artifacts/shap_explainer.pkl')

def explain_prediction(feature_vector: dict, feature_names: list) -> dict:
    """
    feature_vector: dict of {feature_name: value}
    Returns: dict of {feature_name: shap_value} sorted by absolute impact
    """
    X = pd.DataFrame([feature_vector])[feature_names]
    shap_values = explainer.shap_values(X)

    explanation = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, shap_values[0])
    }
    # Sort by absolute contribution descending
    return dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))