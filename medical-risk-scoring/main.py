"""
main.py
医疗用户分析与风险评分系统 – 完整 Pipeline 入口

运行方式：
  python generate_sample_data.py   # 先生成数据
  python main.py                    # 运行完整 pipeline
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split

# 将 src 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import MedicalDataPreprocessor
from feature_engineering import MedicalFeatureEngineer
from model_training import MedicalRiskModelTrainer
from model_evaluation import ModelEvaluator
from rfm_analysis import RFMAnalyzer
from strategy import BusinessStrategyEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("reports/pipeline.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def run_pipeline(task: str = "churn"):
    """
    运行完整 Pipeline
    task: 'churn'（流失风险）或 'revisit'（复诊概率）
    """
    print("\n" + "=" * 70)
    print(f"🏥 医疗用户分析与风险评分系统")
    print(f"   任务：{'患者流失风险预测' if task == 'churn' else '复诊概率预测'}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1：加载数据
    # ----------------------------------------------------------------
    data_path = "data/raw/medical_users.csv"
    if not os.path.exists(data_path):
        print("❌ 原始数据不存在，请先运行：python generate_sample_data.py")
        sys.exit(1)

    df_raw = pd.read_csv(data_path)
    logger.info(f"Step 1 – 数据加载完成：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

    # ----------------------------------------------------------------
    # Step 2：数据清洗
    # ----------------------------------------------------------------
    print("\n[Step 2] 数据清洗与标准化...")
    preprocessor = MedicalDataPreprocessor(iqr_factor=1.5)
    df_clean = preprocessor.fit_transform(df_raw.copy())
    preprocessor.report(df_raw, df_clean)

    df_clean.to_csv("data/processed/medical_users_clean.csv", index=False)
    logger.info("Step 2 – 数据清洗完成")

    # ----------------------------------------------------------------
    # Step 3：RFM 分析（在特征工程之前，便于可视化）
    # ----------------------------------------------------------------
    print("\n[Step 3] RFM 患者分层分析...")
    # 先做基础特征工程（含 RFM）以便分析
    fe_temp = MedicalFeatureEngineer()
    df_rfm_check = fe_temp.fit_transform(df_clean.copy())

    rfm = RFMAnalyzer()
    rfm.plot_rfm_distribution(df_rfm_check)
    rfm.plot_segment_pie(df_rfm_check)
    rfm.segment_summary(df_rfm_check)
    logger.info("Step 3 – RFM 分析完成")

    # ----------------------------------------------------------------
    # Step 4：特征工程
    # ----------------------------------------------------------------
    print("\n[Step 4] 特征工程与用户画像构建...")
    feature_engineer = MedicalFeatureEngineer()
    df_features = feature_engineer.fit_transform(df_clean.copy())
    logger.info(f"Step 4 – 特征工程完成，最终维度：{df_features.shape}")

    # ----------------------------------------------------------------
    # Step 5：数据划分
    # ----------------------------------------------------------------
    print("\n[Step 5] 划分训练集与测试集（80/20）...")
    trainer = MedicalRiskModelTrainer(task=task)
    X, y = trainer.prepare_data(df_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Step 5 – 训练集：{X_train.shape[0]}，测试集：{X_test.shape[0]}")
    print(f"   训练集：{X_train.shape[0]} 样本，测试集：{X_test.shape[0]} 样本")
    print(f"   正样本比例（训练）：{y_train.mean():.1%}")

    # ----------------------------------------------------------------
    # Step 6：模型训练
    # ----------------------------------------------------------------
    print("\n[Step 6] 训练模型（Logistic Regression / Random Forest / XGBoost）...")
    trainer.fit(X_train, y_train)
    trainer.save_models()
    logger.info("Step 6 – 模型训练完成")

    # ----------------------------------------------------------------
    # Step 7：K-Fold 交叉验证
    # ----------------------------------------------------------------
    print("\n[Step 7] K-Fold 交叉验证（k=5）...")
    evaluator = ModelEvaluator(save_dir="reports/figures")
    cv_results = evaluator.cross_validate_all(trainer.models, X_train, y_train, k=5)
    evaluator.plot_model_comparison(cv_results, task=task)

    # ----------------------------------------------------------------
    # Step 8：GridSearchCV 超参数调优（XGBoost）
    # ----------------------------------------------------------------
    print("\n[Step 8] GridSearchCV 超参数调优（XGBoost）...")
    best_xgb = evaluator.tune_xgboost(X_train, y_train)
    trainer.models["XGBoost (Tuned)"] = best_xgb
    logger.info("Step 8 – 超参数调优完成")

    # ----------------------------------------------------------------
    # Step 9：测试集最终评估
    # ----------------------------------------------------------------
    print("\n[Step 9] 测试集最终评估...")
    test_results = evaluator.evaluate_on_test(
        trainer.models, X_test, y_test, scaler=trainer.scaler, task=task
    )

    # 混淆矩阵（最优模型）
    evaluator.plot_confusion_matrix(
        trainer.models["XGBoost (Tuned)"], X_test, y_test,
        model_name="XGBoost Tuned", task=task
    )

    # 特征重要性
    evaluator.plot_feature_importance(
        trainer.models["XGBoost (Tuned)"],
        feature_names=X_train.columns.tolist(),
        top_n=20, task=task
    )

    # ----------------------------------------------------------------
    # Step 10：业务策略输出
    # ----------------------------------------------------------------
    print("\n[Step 10] 业务策略分析与输出...")
    strategy = BusinessStrategyEngine()

    # 使用最优模型对全量数据打分
    X_all, _ = trainer.prepare_data(df_features)
    _, proba_all = trainer.predict(X_all, model_name="XGBoost (Tuned)"
                                    if "XGBoost (Tuned)" in trainer.models else "XGBoost")

    patient_report = strategy.generate_patient_report(df_features, proba_all, task=task)
    strategy.print_intervention_strategies()
    strategy.plot_risk_distribution(proba_all, task=task)

    # 保存患者风险报告
    report_path = f"reports/patient_risk_report_{task}.csv"
    patient_report.to_csv(report_path, index=False)
    print(f"\n  患者风险报告已保存：{report_path}")

    # ----------------------------------------------------------------
    # 完成
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ Pipeline 全部完成！")
    print(f"   📊 可视化图表：reports/figures/")
    print(f"   📋 患者风险报告：{report_path}")
    print(f"   🤖 训练模型：models/")
    print(f"   📝 运行日志：reports/pipeline.log")
    print("=" * 70)

    return test_results, patient_report


if __name__ == "__main__":
    # 运行流失风险预测
    run_pipeline(task="churn")

    print("\n" + "-" * 70)
    print("运行复诊概率预测任务...")
    run_pipeline(task="revisit")
