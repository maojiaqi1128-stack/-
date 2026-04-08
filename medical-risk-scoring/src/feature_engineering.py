"""
src/feature_engineering.py
特征工程与用户画像构建模块
- 业务特征：就诊频率、复诊间隔、慢病标签、用药依从性
- RFM 患者分层
- One-Hot 编码（类别变量）
- Log 变换（偏态特征）
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MedicalFeatureEngineer:
    """医疗特征工程"""

    # 需要 log 变换的偏态特征
    LOG_TRANSFORM_COLS = ["total_medical_fee", "visit_count", "avg_revisit_interval_days"]

    # 需要 One-Hot 的类别特征
    OHE_COLS = ["gender", "region", "insurance_type", "chronic_disease",
                "primary_department"]

    def __init__(self):
        self.ohe_columns_ = None       # 训练后的 OHE 列名
        self._is_fitted = False

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._build_business_features(df)
        df = self._rfm_segmentation(df)
        df = self._log_transform(df)
        df, self.ohe_columns_ = self._one_hot_encode(df, fit=True)
        self._is_fitted = True
        logger.info(f"特征工程完成（fit_transform），最终特征数：{df.shape[1]}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit_transform()")
        df = df.copy()
        df = self._build_business_features(df)
        df = self._rfm_segmentation(df)
        df = self._log_transform(df)
        df, _ = self._one_hot_encode(df, fit=False)
        # 对齐训练时的列
        for col in self.ohe_columns_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.ohe_columns_]
        return df

    # ------------------------------------------------------------------
    # 特征构建
    # ------------------------------------------------------------------

    def _build_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于医疗业务逻辑构建特征"""

        # 1. 就诊频率（每月就诊次数）
        df["visit_frequency_monthly"] = df["visit_count"] / (
            df["registration_months"].clip(1) 
        )

        # 2. 慢病标签（二值化）
        df["has_chronic_disease"] = (df["chronic_disease"] != "None").astype(int)

        # 3. 慢病严重程度（None=0, 单病=1, 多病=2）
        df["chronic_severity"] = df["chronic_disease"].map(
            lambda x: 2 if x == "Multiple" else (0 if x == "None" else 1)
        )

        # 4. 用药依从性等级（Low/Medium/High）
        df["adherence_level"] = pd.cut(
            df["medication_adherence_score"],
            bins=[-0.001, 0.4, 0.7, 1.001],
            labels=["Low", "Medium", "High"]
        ).astype(str)

        # 5. 是否高龄患者（≥65岁）
        df["is_elderly"] = (df["age"] >= 65).astype(int)

        # 6. 年龄分组
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 75, 100],
            labels=["18-30", "31-45", "46-60", "61-75", "76+"]
        ).astype(str)

        # 7. 人均就诊费用
        df["avg_fee_per_visit"] = df["total_medical_fee"] / df["visit_count"].clip(1)

        # 8. 复诊间隔是否规律（变异系数低 = 更规律）
        # 用随访完成率作为规律性代理指标
        df["followup_regularity"] = df["followup_completion_rate"].fillna(0)

        # 9. 综合健康风险指数
        df["health_risk_index"] = (
            df["chronic_severity"] * 0.4 +
            (1 - df["medication_adherence_score"].clip(0, 1)) * 0.3 +
            (1 - df["followup_completion_rate"].clip(0, 1)) * 0.3
        )

        # 10. 是否使用在线问诊
        df["uses_online_consult"] = df["online_consult_count"].clip(0, 1).astype(int)

        logger.info("业务特征构建完成（10 个新特征）")
        return df

    def _rfm_segmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RFM 患者分层
        R (Recency)：最近就诊距今天数（越小越好）
        F (Frequency)：总就诊次数
        M (Monetary)：总就诊费用
        """
        # 分数：5分制，越高越好
        df["R_score"] = pd.qcut(
            df["last_visit_days_ago"], q=5,
            labels=[5, 4, 3, 2, 1], duplicates="drop"
        ).astype(float)

        df["F_score"] = pd.qcut(
            df["visit_count"], q=5,
            labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(float)

        df["M_score"] = pd.qcut(
            df["total_medical_fee"], q=5,
            labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(float)

        df["RFM_score"] = df["R_score"] + df["F_score"] + df["M_score"]

        # 患者分层
        df["rfm_segment"] = pd.cut(
            df["RFM_score"],
            bins=[0, 6, 9, 12, 15.1],
            labels=["Low Value", "Medium Value", "High Value", "Champion"]
        ).astype(str)

        logger.info("RFM 分层完成")
        return df

    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """对偏态特征进行 log(1+x) 变换"""
        for col in self.LOG_TRANSFORM_COLS:
            if col in df.columns:
                new_col = f"log_{col}"
                df[new_col] = np.log1p(df[col].clip(0))
                logger.info(f"  Log 变换：{col} → {new_col}")
        return df

    def _one_hot_encode(self, df: pd.DataFrame, fit: bool = True):
        """One-Hot 编码类别变量"""
        cols_to_encode = [c for c in self.OHE_COLS + ["adherence_level", "age_group", "rfm_segment"]
                          if c in df.columns]

        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)

        if fit:
            ohe_columns = df_encoded.columns.tolist()
            logger.info(f"One-Hot 编码完成，编码列：{cols_to_encode}")
        else:
            ohe_columns = None

        return df_encoded, ohe_columns

    def get_feature_names(self, df: pd.DataFrame) -> list:
        """返回最终特征列名（排除 ID 和标签列）"""
        exclude = ["patient_id", "churn_risk", "revisit_probability"]
        return [c for c in df.columns if c not in exclude]
