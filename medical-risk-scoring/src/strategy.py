"""
src/strategy.py
数据驱动业务策略模块
- 高风险患者识别与分级
- 个性化干预策略输出
- 监控指标体系计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False


class BusinessStrategyEngine:
    """业务策略引擎"""

    RISK_THRESHOLDS = {
        "Critical": 0.75,   # 极高风险
        "High": 0.55,        # 高风险
        "Medium": 0.35,      # 中风险
        "Low": 0.0,          # 低风险
    }

    INTERVENTION_STRATEGIES = {
        "Critical": {
            "label": "极高风险 – 立即干预",
            "actions": [
                "24小时内电话随访，了解未就诊原因",
                "安排专科医生主动外拨，提供免费复诊名额",
                "推送个性化健康提醒（用药提醒、慢病管理）",
                "优先安排绿色通道，减少就诊等待时间",
            ],
            "kpi": ["7日内复诊率", "随访接听率", "紧急干预转化率"],
        },
        "High": {
            "label": "高风险 – 主动触达",
            "actions": [
                "3日内发送复诊提醒短信/APP推送",
                "提供在线问诊优惠券或复诊优惠",
                "推荐慢病管理课程或健康讲座",
                "安排护士随访评估当前健康状态",
            ],
            "kpi": ["30日内复诊率", "推送点击率", "优惠券核销率"],
        },
        "Medium": {
            "label": "中风险 – 定期维护",
            "actions": [
                "每月定期发送健康资讯与复诊提醒",
                "推荐年度体检套餐或健康监测服务",
                "邀请参与患者社群与健康管理活动",
            ],
            "kpi": ["季度复诊率", "健康内容打开率", "体检转化率"],
        },
        "Low": {
            "label": "低风险 – 常规维护",
            "actions": [
                "季度健康简报推送",
                "年度满意度调研",
                "会员积分活动激励",
            ],
            "kpi": ["年度留存率", "满意度评分", "NPS净推荐值"],
        },
    }

    def assign_risk_level(self, proba: np.ndarray) -> pd.Series:
        """根据预测概率分配风险等级"""
        levels = []
        for p in proba:
            if p >= self.RISK_THRESHOLDS["Critical"]:
                levels.append("Critical")
            elif p >= self.RISK_THRESHOLDS["High"]:
                levels.append("High")
            elif p >= self.RISK_THRESHOLDS["Medium"]:
                levels.append("Medium")
            else:
                levels.append("Low")
        return pd.Series(levels)

    def generate_patient_report(self, df: pd.DataFrame, proba: np.ndarray,
                                 task: str = "churn") -> pd.DataFrame:
        """生成患者风险报告"""
        report = pd.DataFrame()
        if "patient_id" in df.columns:
            report["patient_id"] = df["patient_id"].values
        report["churn_probability"] = proba.round(4)
        report["risk_level"] = self.assign_risk_level(proba).values

        # 附加关键特征
        for col in ["age", "chronic_disease", "visit_count", "last_visit_days_ago",
                    "medication_adherence_score", "rfm_segment"]:
            if col in df.columns:
                report[col] = df[col].values

        report = report.sort_values("churn_probability", ascending=False).reset_index(drop=True)

        # 打印分级统计
        print(f"\n{'='*60}")
        print(f"患者风险分级报告 – {task.title()} Prediction")
        print(f"{'='*60}")
        level_counts = report["risk_level"].value_counts()
        total = len(report)
        for level in ["Critical", "High", "Medium", "Low"]:
            count = level_counts.get(level, 0)
            info = self.INTERVENTION_STRATEGIES[level]
            print(f"  {info['label']:<30} {count:>5} 人  ({count/total:.1%})")
        print(f"{'='*60}")

        return report

    def print_intervention_strategies(self):
        """打印完整干预策略"""
        print("\n" + "=" * 60)
        print("数据驱动干预策略体系")
        print("=" * 60)
        for level, info in self.INTERVENTION_STRATEGIES.items():
            print(f"\n🎯 {info['label']}")
            print("   干预措施：")
            for action in info["actions"]:
                print(f"     • {action}")
            print(f"   KPI 监控：{' / '.join(info['kpi'])}")

    def compute_monitoring_metrics(self, df_before: pd.DataFrame,
                                    df_after: pd.DataFrame) -> dict:
        """
        计算业务监控指标
        df_before: 干预前数据
        df_after:  干预后数据（模拟）
        """
        metrics = {}

        if "churn_risk" in df_before.columns:
            metrics["retention_rate_before"] = 1 - df_before["churn_risk"].mean()
        if "churn_risk" in df_after.columns:
            metrics["retention_rate_after"] = 1 - df_after["churn_risk"].mean()

        if "revisit_probability" in df_before.columns:
            metrics["revisit_rate_before"] = df_before["revisit_probability"].mean()
        if "revisit_probability" in df_after.columns:
            metrics["revisit_rate_after"] = df_after["revisit_probability"].mean()

        print("\n📈 业务监控指标：")
        for k, v in metrics.items():
            print(f"   {k}: {v:.1%}")

        return metrics

    def plot_risk_distribution(self, proba: np.ndarray, task: str = "churn",
                                save_dir: str = "reports/figures"):
        """风险概率分布图"""
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # 概率分布直方图
        axes[0].hist(proba, bins=40, color="#3498db", alpha=0.8, edgecolor="white")
        for threshold in [0.35, 0.55, 0.75]:
            axes[0].axvline(threshold, color="#e74c3c", linestyle="--", lw=1.5, alpha=0.7)
        axes[0].set_xlabel("Risk Probability", fontsize=12)
        axes[0].set_ylabel("Patient Count", fontsize=12)
        axes[0].set_title(f"Risk Probability Distribution – {task.title()}", fontsize=12, fontweight="bold")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # 风险等级分布
        risk_levels = self.assign_risk_level(proba)
        level_counts = risk_levels.value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        )
        colors_bar = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        axes[1].bar(level_counts.index, level_counts.values,
                    color=colors_bar, alpha=0.85, edgecolor="white")
        axes[1].set_xlabel("Risk Level", fontsize=12)
        axes[1].set_ylabel("Patient Count", fontsize=12)
        axes[1].set_title("Patient Risk Level Distribution", fontsize=12, fontweight="bold")
        for i, (label, count) in enumerate(level_counts.items()):
            axes[1].text(i, count + 5, str(count), ha="center", fontsize=10, fontweight="bold")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        path = os.path.join(save_dir, f"risk_distribution_{task}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  风险分布图已保存：{path}")
