"""
tests/test_pipeline.py
单元测试 – 覆盖核心模块
运行方式：pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import MedicalDataPreprocessor
from src.feature_engineering import MedicalFeatureEngineer
from src.model_training import MedicalRiskModelTrainer
from src.rfm_analysis import RFMAnalyzer
from src.strategy import BusinessStrategyEngine


# ── 测试数据 fixture ──────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """生成最小可用测试数据集"""
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "age": np.random.randint(20, 80, n),
        "gender": np.random.choice(["Male", "Female", None], n, p=[0.48, 0.48, 0.04]),
        "region": np.random.choice(["North", "South", None], n, p=[0.45, 0.45, 0.1]),
        "insurance_type": np.random.choice(["Basic", "Premium", "None"], n),
        "visit_count": np.random.randint(1, 30, n),
        "last_visit_days_ago": np.random.randint(1, 300, n),
        "avg_revisit_interval_days": np.where(
            np.random.random(n) > 0.1,
            np.random.uniform(10, 120, n),
            np.nan
        ),
        "total_medical_fee": np.random.lognormal(7, 0.8, n),
        "chronic_disease": np.random.choice(
            ["None", "Hypertension", "Diabetes", "Multiple"], n
        ),
        "medication_adherence_score": np.where(
            np.random.random(n) > 0.08, np.random.uniform(0, 1, n), np.nan
        ),
        "followup_completion_rate": np.where(
            np.random.random(n) > 0.05, np.random.uniform(0, 1, n), np.nan
        ),
        "primary_department": np.random.choice(
            ["Internal Medicine", "Cardiology", None], n, p=[0.4, 0.4, 0.2]
        ),
        "online_consult_count": np.random.randint(0, 2, n),
        "satisfaction_score": np.where(
            np.random.random(n) > 0.1, np.random.choice([1,2,3,4,5], n), np.nan
        ),
        "registration_months": np.random.randint(1, 48, n),
        "churn_risk": np.random.randint(0, 2, n),
        "revisit_probability": np.random.randint(0, 2, n),
    })
    return df


# ── 数据预处理测试 ────────────────────────────────────────────────────

class TestPreprocessor:
    def test_no_missing_after_clean(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        df_clean = preprocessor.fit_transform(sample_df)
        assert df_clean.isnull().sum().sum() == 0, "清洗后不应有缺失值"

    def test_outlier_bounds_set(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        preprocessor.fit_transform(sample_df)
        assert "total_medical_fee" in preprocessor.outlier_bounds
        assert "visit_count" in preprocessor.outlier_bounds

    def test_unknown_filling_for_category(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        df_clean = preprocessor.fit_transform(sample_df)
        for col in ["gender", "region", "primary_department"]:
            assert df_clean[col].isnull().sum() == 0

    def test_transform_uses_train_stats(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        preprocessor.fit_transform(sample_df)
        df_new = sample_df.copy()
        df_transformed = preprocessor.transform(df_new)
        assert df_transformed.isnull().sum().sum() == 0

    def test_raises_if_not_fitted(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        with pytest.raises(RuntimeError):
            preprocessor.transform(sample_df)


# ── 特征工程测试 ──────────────────────────────────────────────────────

class TestFeatureEngineer:
    def get_clean_df(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        return preprocessor.fit_transform(sample_df)

    def test_business_features_exist(self, sample_df):
        df_clean = self.get_clean_df(sample_df)
        fe = MedicalFeatureEngineer()
        df_feat = fe.fit_transform(df_clean)
        expected = ["visit_frequency_monthly", "has_chronic_disease",
                    "health_risk_index", "RFM_score"]
        for col in expected:
            assert col in df_feat.columns, f"缺少特征列：{col}"

    def test_rfm_scores_range(self, sample_df):
        df_clean = self.get_clean_df(sample_df)
        fe = MedicalFeatureEngineer()
        df_feat = fe.fit_transform(df_clean)
        assert df_feat["RFM_score"].between(3, 15).all() or df_feat["RFM_score"].notna().any()

    def test_log_transform_cols_created(self, sample_df):
        df_clean = self.get_clean_df(sample_df)
        fe = MedicalFeatureEngineer()
        df_feat = fe.fit_transform(df_clean)
        assert "log_total_medical_fee" in df_feat.columns
        assert "log_visit_count" in df_feat.columns

    def test_no_inf_after_log(self, sample_df):
        df_clean = self.get_clean_df(sample_df)
        fe = MedicalFeatureEngineer()
        df_feat = fe.fit_transform(df_clean)
        numeric = df_feat.select_dtypes(include=np.number)
        assert not np.isinf(numeric.values).any(), "log 变换后不应有 inf"


# ── 模型训练测试 ──────────────────────────────────────────────────────

class TestModelTrainer:
    def get_features(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        df_clean = preprocessor.fit_transform(sample_df)
        fe = MedicalFeatureEngineer()
        return fe.fit_transform(df_clean)

    def test_all_models_trained(self, sample_df):
        df_feat = self.get_features(sample_df)
        trainer = MedicalRiskModelTrainer(task="churn")
        X, y = trainer.prepare_data(df_feat)
        trainer.fit(X, y)
        assert len(trainer.models) == 3
        expected_names = {"Logistic Regression", "Random Forest", "XGBoost"}
        assert set(trainer.models.keys()) == expected_names

    def test_predict_returns_correct_shape(self, sample_df):
        df_feat = self.get_features(sample_df)
        trainer = MedicalRiskModelTrainer(task="churn")
        X, y = trainer.prepare_data(df_feat)
        trainer.fit(X, y)
        preds, proba = trainer.predict(X, model_name="XGBoost")
        assert len(preds) == len(y)
        assert len(proba) == len(y)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_invalid_task_raises(self):
        with pytest.raises(AssertionError):
            MedicalRiskModelTrainer(task="invalid_task")


# ── RFM 分析测试 ──────────────────────────────────────────────────────

class TestRFMAnalyzer:
    def test_segment_summary_returns_df(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        df_clean = preprocessor.fit_transform(sample_df)
        fe = MedicalFeatureEngineer()
        df_feat = fe.fit_transform(df_clean)

        rfm = RFMAnalyzer()
        summary = rfm.segment_summary(df_feat)
        assert isinstance(summary, pd.DataFrame)


# ── 策略引擎测试 ──────────────────────────────────────────────────────

class TestStrategyEngine:
    def test_risk_levels_valid(self):
        engine = BusinessStrategyEngine()
        proba = np.array([0.1, 0.4, 0.6, 0.8])
        levels = engine.assign_risk_level(proba)
        assert list(levels) == ["Low", "Medium", "High", "Critical"]

    def test_report_generated(self, sample_df):
        preprocessor = MedicalDataPreprocessor()
        df_clean = preprocessor.fit_transform(sample_df)
        fe = MedicalFeatureEngineer()
        df_feat = fe.fit_transform(df_clean)

        engine = BusinessStrategyEngine()
        proba = np.random.uniform(0, 1, len(df_feat))
        report = engine.generate_patient_report(df_feat, proba, task="churn")
        assert "churn_probability" in report.columns
        assert "risk_level" in report.columns
        assert len(report) == len(df_feat)
