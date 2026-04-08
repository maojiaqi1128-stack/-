"""
src/model_evaluation.py
模型评估与超参数调优模块
- K-Fold 交叉验证（k=5）
- GridSearchCV 调优（XGBoost）
- AUC / F1 / Recall / Precision 评估
- 可视化：ROC 曲线、混淆矩阵、特征重要性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import logging

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    accuracy_score, confusion_matrix, roc_curve, classification_report
)
from xgboost import XGBClassifier

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估与调优"""

    def __init__(self, save_dir: str = "reports/figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # K-Fold 交叉验证
    # ------------------------------------------------------------------

    def cross_validate_all(self, models: dict, X: pd.DataFrame, y: pd.Series, k: int = 5) -> pd.DataFrame:
        """对所有模型进行 K-Fold 交叉验证"""
        results = []
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for name, model in models.items():
            scoring = {
                "auc": "roc_auc",
                "f1": "f1",
                "recall": "recall",
                "precision": "precision",
            }
            cv_results = cross_validate(
                model, X, y,
                cv=kf,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False
            )

            row = {
                "Model": name,
                "AUC (mean)": cv_results["test_auc"].mean(),
                "AUC (std)": cv_results["test_auc"].std(),
                "F1 (mean)": cv_results["test_f1"].mean(),
                "Recall (mean)": cv_results["test_recall"].mean(),
                "Precision (mean)": cv_results["test_precision"].mean(),
            }
            results.append(row)
            logger.info(
                f"  {name}: AUC={row['AUC (mean)']:.4f}±{row['AUC (std)']:.4f} | "
                f"F1={row['F1 (mean)']:.4f} | Recall={row['Recall (mean)']:.4f}"
            )

        df_results = pd.DataFrame(results).round(4)
        print("\n" + "=" * 70)
        print(f"K-Fold 交叉验证结果（k={k}）")
        print("=" * 70)
        print(df_results.to_string(index=False))
        print("=" * 70)
        return df_results

    # ------------------------------------------------------------------
    # GridSearchCV 超参数调优（XGBoost）
    # ------------------------------------------------------------------

    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """GridSearchCV 调优 XGBoost"""
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "subsample": [0.8, 1.0],
        }

        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )

        grid_search = GridSearchCV(
            xgb, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )

        logger.info("开始 GridSearchCV 超参数调优...")
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        logger.info(f"最优参数：{best_params}")
        logger.info(f"最优 AUC（CV）：{best_score:.4f}")

        print(f"\n✅ GridSearchCV 完成")
        print(f"   最优参数：{best_params}")
        print(f"   最优 AUC（CV）：{best_score:.4f}")

        return grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 测试集评估与可视化
    # ------------------------------------------------------------------

    def evaluate_on_test(self, models: dict, X_test: pd.DataFrame,
                         y_test: pd.Series, scaler=None, task: str = "churn"):
        """在测试集上评估所有模型，生成可视化"""
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        colors = ["#3498db", "#2ecc71", "#e74c3c"]

        report_rows = []

        for (name, model), color in zip(models.items(), colors):
            if name == "Logistic Regression" and scaler is not None:
                X_input = scaler.transform(X_test)
            else:
                X_input = X_test

            y_pred = model.predict(X_input)
            y_prob = model.predict_proba(X_input)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            report_rows.append({
                "Model": name, "AUC": round(auc, 4), "F1": round(f1, 4),
                "Recall": round(rec, 4), "Precision": round(prec, 4), "Accuracy": round(acc, 4)
            })

            # ROC 曲线
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Baseline")
        ax_roc.set_xlabel("False Positive Rate", fontsize=12)
        ax_roc.set_ylabel("True Positive Rate", fontsize=12)
        ax_roc.set_title(f"ROC Curve Comparison – {task.title()} Prediction", fontsize=13, fontweight="bold")
        ax_roc.legend(loc="lower right", fontsize=10)
        ax_roc.spines["top"].set_visible(False)
        ax_roc.spines["right"].set_visible(False)
        ax_roc.grid(alpha=0.3)

        roc_path = os.path.join(self.save_dir, f"roc_curve_{task}.png")
        fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig_roc)
        print(f"  ROC 曲线已保存：{roc_path}")

        # 汇总表
        df_report = pd.DataFrame(report_rows)
        print("\n测试集评估结果：")
        print(df_report.to_string(index=False))

        return df_report

    def plot_confusion_matrix(self, model, X_test, y_test,
                               model_name: str = "XGBoost", task: str = "churn",
                               scaler=None):
        """绘制混淆矩阵"""
        X_input = scaler.transform(X_test) if (model_name == "Logistic Regression" and scaler) else X_test
        y_pred = model.predict(X_input)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"],
            ax=ax
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix – {model_name} ({task})", fontsize=12, fontweight="bold")

        path = os.path.join(self.save_dir, f"confusion_matrix_{model_name.lower().replace(' ','_')}_{task}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  混淆矩阵已保存：{path}")

    def plot_feature_importance(self, model, feature_names: list,
                                 top_n: int = 20, task: str = "churn"):
        """特征重要性（仅适用于树模型）"""
        if not hasattr(model, "feature_importances_"):
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            range(len(indices)),
            importances[indices][::-1],
            color="#3498db", alpha=0.85
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importance – {task.title()} Prediction",
                     fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        path = os.path.join(self.save_dir, f"feature_importance_{task}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  特征重要性图已保存：{path}")

    def plot_model_comparison(self, df_results: pd.DataFrame, task: str = "churn"):
        """模型指标对比柱状图"""
        metrics = ["AUC", "F1", "Recall", "Precision"]
        available_metrics = [m for m in metrics if m + " (mean)" in df_results.columns or m in df_results.columns]

        # 兼容交叉验证结果格式
        col_map = {}
        for m in metrics:
            if m + " (mean)" in df_results.columns:
                col_map[m] = m + " (mean)"
            elif m in df_results.columns:
                col_map[m] = m

        fig, axes = plt.subplots(1, len(col_map), figsize=(14, 5))
        colors = ["#3498db", "#2ecc71", "#e74c3c"]

        for ax, (metric, col) in zip(axes, col_map.items()):
            bars = ax.bar(df_results["Model"], df_results[col],
                          color=colors[:len(df_results)], alpha=0.85, edgecolor="white")
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_xticklabels(df_results["Model"], rotation=15, ha="right", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, val in zip(bars, df_results[col]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        fig.suptitle(f"Model Comparison – {task.title()} Prediction", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"model_comparison_{task}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  模型对比图已保存：{path}")
