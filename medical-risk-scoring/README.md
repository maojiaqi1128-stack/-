# 医疗用户分析与风险评分系统

**Medical User Analysis & Risk Scoring System (Capstone)**  
📅 2025.9 – 2025.12 | 独立项目

---

## 项目简介

本项目基于医疗用户行为数据（就诊记录、随访数据、用药信息等），构建患者流失风险与复诊概率预测模型，帮助医疗机构识别高风险患者群体，制定精准干预策略，提升患者留存率与医疗服务质量。

## 技术栈

- **数据处理**：Python 3.10 / Pandas / NumPy
- **机器学习**：Scikit-learn / XGBoost
- **模型评估**：AUC / F1-score / K-Fold Cross Validation
- **可视化**：Matplotlib / Seaborn
- **环境管理**：pip / requirements.txt

---

## 项目结构

```
medical-risk-scoring/
├── data/
│   ├── raw/                    # 原始数据（不上传，见 .gitignore）
│   └── processed/              # 清洗后数据
├── notebooks/
│   └── exploratory_analysis.ipynb   # 探索性分析
├── src/
│   ├── data_preprocessing.py   # 数据清洗与标准化
│   ├── feature_engineering.py  # 特征工程与用户画像
│   ├── model_training.py       # 模型构建与训练
│   ├── model_evaluation.py     # 模型评估与调参
│   ├── rfm_analysis.py         # RFM 患者分层
│   └── strategy.py             # 业务策略输出
├── models/                     # 保存训练好的模型
├── reports/
│   └── figures/                # 可视化图表
├── tests/
│   └── test_pipeline.py        # 单元测试
├── main.py                     # 主入口，运行完整 pipeline
├── generate_sample_data.py     # 生成模拟数据
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/medical-risk-scoring.git
cd medical-risk-scoring
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 生成模拟数据

```bash
python generate_sample_data.py
```

### 4. 运行完整 Pipeline

```bash
python main.py
```

### 5. 查看结果

训练结果与可视化图表保存在 `reports/figures/`，模型文件保存在 `models/`。

---

## 核心模块说明

### 数据清洗与标准化 (`src/data_preprocessing.py`)
- 数值型缺失值：**中位数填补**
- 类别型缺失值：**"Unknown" 标记**
- 异常值识别：**IQR 方法**（异常费用、就诊频次）
- 数据类型标准化与一致性校验

### 特征工程 (`src/feature_engineering.py`)
- 就诊频率、复诊间隔、慢病标签、用药依从性
- RFM 模型患者分层（Recency / Frequency / Monetary）
- One-Hot 编码（类别变量）
- Log 变换（偏态特征）

### 模型构建 (`src/model_training.py`)
| 模型 | 用途 |
|------|------|
| Logistic Regression | Baseline 基线模型 |
| Random Forest | 非线性关系捕捉 |
| XGBoost | 最优预测性能 |

### 模型评估 (`src/model_evaluation.py`)
- **K-Fold 交叉验证**（k=5）评估稳定性
- **GridSearchCV** 超参数调优（max_depth / learning_rate / n_estimators）
- **AUC、F1-score、Recall** 多维评估，重点优化高风险患者召回率

### 业务策略 (`src/strategy.py`)
- 高流失风险患者识别与分级
- 个性化干预策略（随访、健康提醒、复诊推荐）
- 监控指标体系（留存率、复诊率、转化率）

---

## 模型效果（模拟数据）

| 模型 | AUC | F1-Score | Recall (高风险) |
|------|-----|----------|----------------|
| Logistic Regression | ~0.78 | ~0.72 | ~0.68 |
| Random Forest | ~0.85 | ~0.79 | ~0.76 |
| XGBoost | ~0.88 | ~0.83 | ~0.81 |

---

## 数据说明

> ⚠️ 原始数据涉及患者隐私，不纳入版本控制。  
> 运行 `generate_sample_data.py` 可生成符合同等分布的**模拟数据**用于演示与测试。

---

## 业务价值

- 🎯 精准识别高流失风险患者，提前介入降低漏诊率
- 📊 数据驱动的患者分层管理，优化医疗资源配置
- 🔄 可复用的 Pipeline，支持新数据的滚动更新与再训练

---

## 作者

**[Your Name]**  
Email: your_email@example.com  
GitHub: [your-username](https://github.com/your-username)
