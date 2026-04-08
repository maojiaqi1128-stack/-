"""
src/rfm_analysis.py
RFM 患者分层分析与可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


class RFMAnalyzer:
    """RFM 患者分层分析"""

    SEGMENT_COLORS = {
        "Champion": "#2ecc71",
        "High Value": "#3498db",
        "Medium Value": "#f39c12",
        "Low Value": "#e74c3c",
    }

    def plot_rfm_distribution(self, df: pd.DataFrame, save_dir: str = "reports/figures"):
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("RFM Score Distribution", fontsize=14, fontweight="bold")

        for ax, col, title, color in zip(
            axes,
            ["R_score", "F_score", "M_score"],
            ["Recency Score", "Frequency Score", "Monetary Score"],
            ["#e74c3c", "#3498db", "#2ecc71"]
        ):
            if col in df.columns:
                counts = df[col].value_counts().sort_index()
                ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.8, edgecolor="white")
                ax.set_title(title, fontsize=12)
                ax.set_xlabel("Score")
                ax.set_ylabel("Patient Count")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(save_dir, "rfm_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  RFM 分布图已保存：{path}")

    def plot_segment_pie(self, df: pd.DataFrame, save_dir: str = "reports/figures"):
        os.makedirs(save_dir, exist_ok=True)

        if "rfm_segment" not in df.columns:
            return

        seg_counts = df["rfm_segment"].value_counts()
        colors = [self.SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_counts.index]

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            seg_counts.values,
            labels=seg_counts.index,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for text in autotexts:
            text.set_fontsize(10)
            text.set_fontweight("bold")

        ax.set_title("Patient RFM Segmentation", fontsize=14, fontweight="bold", pad=20)
        path = os.path.join(save_dir, "rfm_segments.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  RFM 分层饼图已保存：{path}")

    def segment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """各分层患者统计摘要"""
        if "rfm_segment" not in df.columns:
            return pd.DataFrame()

        summary = df.groupby("rfm_segment").agg(
            patient_count=("patient_id", "count") if "patient_id" in df.columns else ("R_score", "count"),
            avg_visit_count=("visit_count", "mean"),
            avg_fee=("total_medical_fee", "mean"),
            churn_rate=("churn_risk", "mean"),
            revisit_rate=("revisit_probability", "mean"),
        ).round(3)

        print("\n📊 RFM 分层摘要：")
        print(summary.to_string())
        return summary
