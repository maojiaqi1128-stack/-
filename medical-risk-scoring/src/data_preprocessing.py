"""
src/data_preprocessing.py
数据清洗与标准化模块
- 数值型缺失值：中位数填补
- 类别型缺失值："Unknown" 标记
- 异常值识别与处理：IQR 方法
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MedicalDataPreprocessor:
    """医疗数据清洗与标准化"""

    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor
        self.numeric_medians = {}       # 保存训练集中位数，供 transform 使用
        self.outlier_bounds = {}        # 保存 IQR 边界
        self._is_fitted = False

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换（用于训练集）"""
        df = df.copy()
        df = self._standardize_dtypes(df)
        df = self._fit_fill_missing(df)
        df = self._fit_handle_outliers(df)
        self._is_fitted = True
        logger.info("数据清洗完成（fit_transform）")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """仅转换，使用已拟合的统计量（用于测试/新数据）"""
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit_transform()")
        df = df.copy()
        df = self._standardize_dtypes(df)
        df = self._transform_fill_missing(df)
        df = self._transform_handle_outliers(df)
        logger.info("数据清洗完成（transform）")
        return df

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _standardize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据类型"""
        # 去除首尾空格（字符串列）
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

        logger.info(f"字符串标准化完成，涉及列：{list(str_cols)}")
        return df

    def _fit_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合缺失值填补策略并填补"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        # 数值型：中位数填补
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                self.numeric_medians[col] = median_val
                missing_count = df[col].isnull().sum()
                df[col] = df[col].fillna(median_val)
                logger.info(f"  [数值填补] {col}: {missing_count} 个缺失 → 中位数 {median_val:.4f}")

        # 类别型："Unknown" 标记
        for col in cat_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                df[col] = df[col].fillna("Unknown")
                logger.info(f"  [类别填补] {col}: {missing_count} 个缺失 → 'Unknown'")

        return df

    def _transform_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用训练集统计量填补缺失值"""
        for col, median_val in self.numeric_medians.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(median_val)

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")

        return df

    def _fit_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IQR 方法识别并修正异常值
        目标列：total_medical_fee、visit_count（就诊费用与频次）
        """
        outlier_cols = [c for c in ["total_medical_fee", "visit_count"] if c in df.columns]

        for col in outlier_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.iqr_factor * IQR
            upper = Q3 + self.iqr_factor * IQR

            self.outlier_bounds[col] = (lower, upper)

            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            logger.info(
                f"  [异常值处理] {col}: 识别 {outliers} 个异常 → "
                f"截断至 [{lower:.2f}, {upper:.2f}]"
            )

        return df

    def _transform_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
        return df

    def report(self, df_before: pd.DataFrame, df_after: pd.DataFrame):
        """打印清洗前后对比报告"""
        print("\n" + "=" * 60)
        print("数据清洗报告")
        print("=" * 60)
        print(f"样本数量：{len(df_before):,} → {len(df_after):,}")
        print(f"\n缺失值对比（处理前）：")
        missing_before = df_before.isnull().sum()
        missing_before = missing_before[missing_before > 0]
        print(missing_before.to_string() if len(missing_before) > 0 else "  无缺失值")
        print(f"\n缺失值对比（处理后）：")
        missing_after = df_after.isnull().sum()
        missing_after = missing_after[missing_after > 0]
        print(missing_after.to_string() if len(missing_after) > 0 else "  ✅ 无缺失值")
        print("=" * 60)
