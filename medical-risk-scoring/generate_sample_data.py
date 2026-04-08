"""
generate_sample_data.py
生成模拟医疗用户数据，用于演示与测试
"""

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N = 2000  # 患者数量


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_data():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)

    patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, N + 1)]

    # 基础人口信息
    ages = np.random.randint(18, 85, N)
    genders = np.random.choice(["Male", "Female"], N, p=[0.48, 0.52])
    regions = np.random.choice(
        ["North", "South", "East", "West", "Central"], N,
        p=[0.2, 0.25, 0.2, 0.15, 0.2]
    )
    insurance_types = np.random.choice(
        ["Basic", "Standard", "Premium", "None"], N,
        p=[0.3, 0.35, 0.2, 0.15]
    )

    # 就诊行为
    visit_counts = np.random.negative_binomial(3, 0.4, N) + 1
    visit_counts = np.clip(visit_counts, 1, 50)

    # 最近一次就诊距今天数（Recency）
    last_visit_days = np.random.exponential(60, N).astype(int)
    last_visit_days = np.clip(last_visit_days, 1, 365)

    # 平均复诊间隔（天）
    avg_revisit_interval = np.where(
        visit_counts > 1,
        np.random.normal(45, 20, N).clip(7, 180),
        np.nan
    )

    # 就诊费用（元）
    base_fee = np.random.lognormal(7.5, 0.8, N)
    # 注入异常值（约 3%）
    outlier_mask = np.random.random(N) < 0.03
    base_fee[outlier_mask] = np.random.uniform(50000, 200000, outlier_mask.sum())
    total_fee = base_fee * visit_counts * np.random.uniform(0.8, 1.2, N)

    # 慢病标签
    chronic_diseases = np.random.choice(
        ["None", "Hypertension", "Diabetes", "Heart Disease", "COPD", "Multiple"],
        N, p=[0.35, 0.25, 0.18, 0.1, 0.05, 0.07]
    )

    # 用药依从性评分 (0-1)
    medication_adherence = np.where(
        chronic_diseases != "None",
        np.random.beta(5, 2, N),
        np.random.beta(3, 1, N)
    )
    # 注入缺失值
    missing_mask = np.random.random(N) < 0.08
    medication_adherence = medication_adherence.astype(float)
    medication_adherence[missing_mask] = np.nan

    # 随访完成率
    followup_rate = np.random.beta(4, 2, N)
    followup_missing = np.random.random(N) < 0.05
    followup_rate[followup_missing] = np.nan

    # 主要就诊科室
    departments = np.random.choice(
        ["Internal Medicine", "Cardiology", "Endocrinology",
         "Orthopedics", "Neurology", "Emergency", "General"],
        N, p=[0.25, 0.15, 0.15, 0.12, 0.1, 0.08, 0.15]
    )

    # 注入类别缺失
    dept_missing = np.random.random(N) < 0.04
    departments = np.where(dept_missing, np.nan, departments)

    # 是否有在线问诊记录
    online_consult = np.random.choice([0, 1], N, p=[0.6, 0.4])

    # 患者满意度评分 (1-5)
    satisfaction = np.random.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.1, 0.2, 0.4, 0.25])
    sat_missing = np.random.random(N) < 0.12
    satisfaction = satisfaction.astype(float)
    satisfaction[sat_missing] = np.nan

    # 注册时长（月）
    registration_months = np.random.randint(1, 60, N)

    # 构建目标变量（流失风险）
    # 流失风险由多个因素综合影响
    churn_score = (
        0.3 * (last_visit_days / 365) +
        0.2 * (1 - followup_rate.clip(0, 1)) +
        0.15 * (1 - medication_adherence.clip(0, 1)) +
        0.15 * ((5 - satisfaction.clip(1, 5)) / 4) +
        0.1 * (1 / (visit_counts + 1)) +
        0.1 * np.random.random(N)
    )
    churn_score = np.nan_to_num(churn_score, nan=0.5)
    churn_label = (churn_score > 0.45).astype(int)

    # 复诊概率目标变量
    revisit_score = (
        0.25 * (visit_counts / visit_counts.max()) +
        0.2 * medication_adherence.clip(0, 1) +
        0.2 * followup_rate.clip(0, 1) +
        0.15 * (satisfaction.clip(1, 5) / 5) +
        0.1 * online_consult +
        0.1 * np.random.random(N)
    )
    revisit_score = np.nan_to_num(revisit_score, nan=0.4)
    revisit_label = (revisit_score > 0.45).astype(int)

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "gender": genders,
        "region": regions,
        "insurance_type": insurance_types,
        "visit_count": visit_counts,
        "last_visit_days_ago": last_visit_days,
        "avg_revisit_interval_days": avg_revisit_interval,
        "total_medical_fee": total_fee.round(2),
        "chronic_disease": chronic_diseases,
        "medication_adherence_score": medication_adherence.round(4),
        "followup_completion_rate": followup_rate.round(4),
        "primary_department": departments,
        "online_consult_count": online_consult,
        "satisfaction_score": satisfaction,
        "registration_months": registration_months,
        "churn_risk": churn_label,
        "revisit_probability": revisit_label,
    })

    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = generate_data()
    df.to_csv("data/raw/medical_users.csv", index=False)

    print(f"✅ 模拟数据生成完毕：{len(df)} 条患者记录")
    print(f"   流失患者比例：{df['churn_risk'].mean():.1%}")
    print(f"   复诊患者比例：{df['revisit_probability'].mean():.1%}")
    print(f"   缺失值概览：")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())
    print(f"\n数据已保存至 data/raw/medical_users.csv")
