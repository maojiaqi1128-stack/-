"""
src/model_training.py
模型构建与训练模块
- Logistic Regression（Baseline）
- Random Forest
- XGBoost
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MedicalRiskModelTrainer:
    """医疗风险预测模型训练"""

    def __init__(self, task: str = "churn"):
        """
        task: 'churn' (流失风险) 或 'revisit' (复诊概率)
        """
        assert task in ("churn", "revisit"), "task 必须为 'churn' 或 'revisit'"
        self.task = task
        self.target_col = "churn_risk" if task == "churn" else "revisit_probability"
        self.scaler = StandardScaler()
        self.models = {}
        self._feature_cols = None

    def prepare_data(self, df: pd.DataFrame):
        """准备特征矩阵与标签"""
        exclude = {"patient_id", "churn_risk", "revisit_probability"}
        self._feature_cols = [c for c in df.columns if c not in exclude
                               and df[c].dtype in [np.float64, np.int64, np.int32, np.float32, int, float]]

        X = df[self._feature_cols].fillna(0).astype(float)
        y = df[self.target_col].astype(int)

        logger.info(f"特征数量：{X.shape[1]}，样本数：{X.shape[0]}")
        return X, y

    def build_models(self) -> dict:
        """构建三个候选模型"""
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
                C=1.0
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=1,
                random_state=42,
                verbosity=0
            ),
        }
        return models

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """训练所有模型"""
        X_scaled = self.scaler.fit_transform(X_train)

        for name, model in self.build_models().items():
            if name == "Logistic Regression":
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)  # 树模型无需标准化
            self.models[name] = model
            logger.info(f"  ✅ {name} 训练完成")

        return self.models

    def predict(self, X: pd.DataFrame, model_name: str = "XGBoost"):
        """预测"""
        model = self.models[model_name]
        if model_name == "Logistic Regression":
            X_input = self.scaler.transform(X)
        else:
            X_input = X
        return model.predict(X_input), model.predict_proba(X_input)[:, 1]

    def save_models(self, save_dir: str = "models"):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self.models.items():
            filename = name.lower().replace(" ", "_") + f"_{self.task}.pkl"
            path = os.path.join(save_dir, filename)
            joblib.dump(model, path)
            logger.info(f"  模型已保存：{path}")

        scaler_path = os.path.join(save_dir, f"scaler_{self.task}.pkl")
        joblib.dump(self.scaler, scaler_path)

        feature_path = os.path.join(save_dir, f"feature_cols_{self.task}.pkl")
        joblib.dump(self._feature_cols, feature_path)
        logger.info(f"  Scaler 和特征列已保存")

    def load_models(self, save_dir: str = "models"):
        """加载模型"""
        for name in ["logistic_regression", "random_forest", "xgboost"]:
            path = os.path.join(save_dir, f"{name}_{self.task}.pkl")
            if os.path.exists(path):
                display_name = name.replace("_", " ").title()
                self.models[display_name] = joblib.load(path)

        scaler_path = os.path.join(save_dir, f"scaler_{self.task}.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        feature_path = os.path.join(save_dir, f"feature_cols_{self.task}.pkl")
        if os.path.exists(feature_path):
            self._feature_cols = joblib.load(feature_path)

    @property
    def feature_cols(self):
        return self._feature_cols
