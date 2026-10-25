# Course Alignment — what we can use from DSML and DL

Source material reviewed (October 2026): DSML course CMP707 (teaching plan, evaluation plan, 12 slide
decks, 16 class notebooks) and Deep Learning Units 1–2 (slides + assessment syllabus). The goal is
to apply techniques the professors actually taught, so the project visibly reflects the courses.

**Key facts**
- DSML evaluation: the project is worth **20 marks** (Option 2) and is *"linked with Project
  Evaluation of the course CMP708"*, so one project serves both subjects.
- DSML has **3 lectures on anomaly detection** (L-34–36), and its slide *"Approaches to anomaly
  detection"* lists: Z-score, Percentiles, **Isolation Forest, One-Class SVM, KNN, Autoencoders**,
  **DBSCAN**, K-Means, Moving Averages, Seasonal Decomposition.
- DL Units 1–2 are about *training* networks well (optimisers, learning rates, initialisation,
  normalisation, regularisation). LSTM/GRU are previewed (Unit 4); autoencoders appear only on the
  DSML anomaly slide.

## Final selection (team decision, October 2026)

Few methods, one clear story per problem: **simple baseline → machine learning → deep learning**.

| Problem | Method | Course | Where it is taught |
|---|---|---|---|
| Driver mistake detection | Z-score (baseline) | DSML | 4.3 Anomaly Detection p6 (statistical methods) |
| | **Isolation Forest** | DSML | 4.3 p6 (ML approaches) |
| | **LSTM autoencoder** | DL (+DSML) | 4.3 p6 (autoencoders); trained with DL Unit 2 practice |
| Tyre / pace prediction | Ridge regression (baseline) | DSML | `LinearReg.ipynb`; metrics 3.1 p10–14 |
| | **Vanilla RNN** | DL | NN deck p61–72 (RNN, BPTT) |
| | **LSTM** | DL | NN deck p74–85 (vanishing gradients; LSTM/GRU as the remedy); U1 p7–8 |

**Evaluation (taught, not extra algorithms):** confusion matrix, precision, recall, F1, specificity
(3.1 p8–16) for mistake detection; MAE, MSE, RMSE, R², MAPE (3.1 p10–14) for pace prediction;
train vs test error for over/underfitting (3.1 p38–39; U2 p34–35).
**Report sections (no new algorithms):** descriptive statistics and correlation (2.1; 3.1 p33–35);
preprocessing choices (3.1 p28–30); CNN theory to explain YOLO (NN deck p30–59).
**Optional, decided later:** one DL training-practice experiment (e.g. optimiser comparison, U2 p7–20);
a 5- vs 10-lap input window for RNN vs LSTM (with 5-lap inputs the two are expected to score closely,
because vanishing gradients mainly affect long sequences).
**Dropped:** One-Class SVM, KNN, DBSCAN, LOF; Random Forest / Gradient Boosting; clustering + PCA;
classification task; GRU.

The tables below are the full survey of the course material, kept for reference.

Legend — **Fit**: ★★★ strong / ★★ good / ★ minor. **Effort**: S / M / L.

---

## 1. Data Science & Machine Learning

### 1A. Anomaly detection — the Isolation Forest rework (roadmap §B)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| Compare detectors **exactly from the professor's list**: Z-score, Isolation Forest, One-Class SVM, KNN (distance to k-th neighbour), DBSCAN (noise points = anomalies) | 4.3 Anomaly Detection p6–9; `DBScan_Demo.ipynb`; `kNN_Test.ipynb`; `SVM_Demo.ipynb` | Same features and planted-mistake test for all; **replaces LOF** (not taught) | ★★★ | M |
| Autoencoder as the deep anomaly detector | 4.3 p6 ("Autoencoders" under ML approaches) | The planned LSTM autoencoder (DL) joins the same comparison | ★★★ | (DL §2C) |
| Evaluate with **confusion matrix, precision, recall, F1, specificity**; explain why accuracy is misleading for rare events | 3.1 ML Basics p8–16, p32 (imbalanced data) | Planted mistakes = labelled positives → full confusion matrix per model; PR-AUC kept as an extra | ★★★ | S |
| Percentile thresholding | 4.3 p6 | Alternative to the contamination setting: flag the top x % of scores | ★★ | S |
| Moving averages | 4.3 p6 | Smooth anomaly scores over time before grouping into mistake events | ★ | S |

### 1B. Regression baselines vs the LSTM (roadmap §C)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| Linear / **Ridge** regression | `LinearReg.ipynb` (Ridge + StandardScaler + train/test split); 4.1 p3 | Baseline for next-lap pace from the previous 5 laps (flattened) | ★★★ | S |
| **Random Forest, Gradient Boosting** (+ Bagging) | `Ensemble_Demo.ipynb`; 1.1 p9 | Non-linear baselines | ★★★ | S |
| Metrics **MAE, MSE, RMSE, R², MAPE** | 3.1 p10–14 (formulas + worked tables) | Report all five for every model incl. the LSTM | ★★★ | S |
| Actual-vs-predicted scatter with ideal line | `LinearReg.ipynb` | One plot per model | ★★ | S |
| Over/underfitting: train vs test error | 3.1 p38–39 | Table of train and test error per model | ★★ | S |

### 1C. Clustering driving styles + PCA (roadmap §F, currently undecided)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| **K-Means** + elbow (WCSS / inertia vs K) | 4.1 p16–26; `KMeans_Demo.ipynb` (elbow loop, from-scratch k-means) | Cluster drivers' per-corner or per-lap style vectors (braking point, min speed, throttle pickup) | ★★★ | M |
| **Hierarchical clustering**, dendrogram, linkage comparison | 4.1 p27–37 | Same vectors → dendrogram of drivers/teams | ★★★ | S |
| **PCA**: standardise first, scree plot, cumulative variance, PC1/PC2 scatter | 4.1 p38–50; 4.2 (worked example); `PCA_Demo.ipynb` | Reduce style vectors to 2-D for visualisation; scree plot in report | ★★★ | S |
| PCA "helps outlier detection" | 4.2 p16 | Optional: PCA reconstruction error as one more anomaly baseline | ★ | S |

### 1D. Classification task (roadmap §F, currently undecided)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| **Logistic Regression, KNN (K sweep), SVM, Naive Bayes, Random Forest, Voting/Stacking** | `LogisticReg`, `kNN_Test`, `SVM_Demo`, `NaiveBayes_Example`, `Ensemble_Demo` notebooks | e.g. predict **tyre compound** (Soft/Medium/Hard) from lap telemetry, or lap type | ★★★ | M |
| Multi-class **confusion matrix**, per-class TP/FP/FN, `classification_report` | 3.1 p13–16; notebooks | Standard evaluation | ★★★ | S |
| One-hot encoding, class imbalance (under/oversampling) | 3.1 p29, p32; `DSML_17Sep2026.ipynb` | Compounds are imbalanced | ★★ | S |
| PCA → classifier pipeline | `PCA_Demo.ipynb` | Optional variant | ★ | S |

### 1E. Statistics, EDA and preprocessing (documentation + a notebook/report section)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| Descriptive statistics: mean/median/mode, SD, IQR, percentiles, **skewness, kurtosis** | 2.1 Descriptive Statistics | Summary table for speed, lap time, throttle per session | ★★★ | S |
| Outliers by **Z-score / IQR**; capping | 3.1 p28; 2.1 p9 | Documents our clipping step and checks it | ★★ | S |
| Correlation heatmap; **Pearson / Spearman / point-biserial**; correlation feature selection (keep if \|r\| ≥ 0.10 with target; drop one of a pair if \|r\| > 0.85) | 3.1 p33–35; 1.5 p4 | Feature selection for the regression/classification work; point-biserial suits the binary Brake flag | ★★★ | S |
| Scaling choices (Standard / Min-Max / Robust) and why | 3.1 p30 | Justify StandardScaler (LSTM) vs fixed ranges (IF) | ★★ | S |
| Missing values (imputation), data-quality dimensions, sensor/IoT data | 3.1 p20–28; `DSML_17Sep2026.ipynb` (SimpleImputer, KNNImputer) | Describe our forward-fill and data quality in the report | ★★ | S |
| Histograms, box plots, scatter, dashboards with **Streamlit/Plotly** | 1.4, 1.5 (name Streamlit and Plotly explicitly) | Our dashboard already is course material | ★★ | — |
| Data-science process (business understanding → … → communication) | 1.1 | Structure of the report | ★ | S |

**Not useful for this project:** recommendation systems (5.1), web-scraping demo (we already
scrape with yt-dlp), NumPy/Pandas basics (already used throughout).

**Taught only by name (we should not over-claim):** train/test split, cross-validation, Lasso,
ROC-AUC/PR-AUC, elbow/silhouette, hypothesis testing — use them where sensible but present the taught
methods first.

---

## 2. Deep Learning

All of these apply to the LSTM (which must be retrained anyway) and to the planned LSTM autoencoder.
Each is shown as a controlled experiment with **loss curves and a results table**.

### 2A. Training-practice experiments on the LSTM (strongest alignment with Units 1–2)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| **Optimiser comparison**: SGD, SGD+Momentum, Nesterov, RMSProp, Adam, AdamW | U2 p7–20 (formulas, evolution map, comparison table; RMSProp "used in RNNs") | Same model and data per optimiser; overlaid validation-loss curves; epochs-to-converge + test MAE table | ★★★ | M |
| **Learning-rate study**: too low / good / too high | U1 p93–96 | Reproduce the slide's loss-vs-epoch chart on our data | ★★★ | S |
| **LR scheduling**: fixed vs Step decay vs Exponential decay vs **Cosine annealing** | U1 p97–99 (cosine formula) | LR-per-epoch plot + loss curves | ★★★ | S |
| **Regularisation**: dropout 0.2/0.3/0.5, **L2 / weight decay**, **early stopping** with patience | U2 p31–33 | Sweep table; mark best epoch on the curve (best-epoch restore already exists) | ★★★ | S |
| **Over/underfitting & bias–variance** diagnostics | U2 p34–35; U1 p103–104 | Train vs validation curves for hidden 32/64/128, labelled under/good/over-fit | ★★★ | S |
| **LayerNorm** (vs none; why not BatchNorm for sequences) | U2 p28–30 | Ablation on the LSTM/autoencoder | ★★ | S |
| **Weight initialisation**: default vs Xavier vs He | U2 p21–27 | Convergence comparison; optional gradient-norm plot (vanishing/exploding) | ★★ | S |
| **Grid search vs random search** (same budget) | U2 p36 | We use grid search; add random search and compare | ★★ | S |
| **Mini-batch size** comparison | U1 p102 | 8 / 32 / 128 table | ★ | S |
| Input normalisation, save best model, monitor validation loss | U2 p37 (best practices) | Already done; cite in the report | ★★ | — |

### 2B. Architecture comparisons
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| **Vanilla RNN vs LSTM vs GRU** — vanishing gradients as the reason for gated cells | NN deck p61–85 (RNN, BPTT, vanishing gradient, LSTM/GRU remedy); U1 p7–8 | Same task, three cells; explains why we use an LSTM | ★★★ | M |
| **MLP** on the flattened 5-lap window | U1 p74–90 (MLP); NN deck p7–28 | Non-recurrent neural baseline | ★★ | S |
| Activation choice (linear output for regression; ReLU/tanh/LeakyReLU in hidden layers) | U1 p57–68 | Justify choices; small ablation in the MLP/autoencoder | ★ | S |

### 2C. LSTM autoencoder (roadmap §D)
| Use | Where taught | How it applies | Fit | Effort |
|---|---|---|---|---|
| Autoencoder for anomaly detection | DSML 4.3 p6 (not in DL Units 1–2) | Reconstruction error = mistake score; joins the anomaly comparison (§1A) | ★★★ | M |
| Its training uses §2A practice (Adam/AdamW, LayerNorm, dropout, early stopping, MSE) | U2 | Same experiment style | ★★ | — |

### 2D. Report / theory only (no code change)
| Use | Where taught | How it applies |
|---|---|---|
| CNN theory: convolution, stride, padding, pooling, segmentation | NN deck p30–59 | Explain how YOLOv11-Seg works (YOLO stays frozen) |
| ML vs DL, the "feature-engineering bottleneck" | U1 p27–28 | Motivates LSTM (learned features) vs Isolation Forest (hand-made features) |
| Forward/backward pass, backpropagation | U2 p5–6; NN deck p19–28 | Describe the LSTM training loop |

**Not useful for this project:** symbolic AI / expert systems, McCulloch–Pitts and logic gates,
softmax/cross-entropy (unless the classification task is adopted), InstanceNorm/GroupNorm,
Transformers (timeline only).

---

## 3. Original suggested priority (superseded by the final selection at the top)

| # | Item | Subject | Why |
|---|---|---|---|
| 1 | Isolation Forest rework with the professor's own anomaly-detection list (Z-score, IF, One-Class SVM, KNN, DBSCAN) + confusion-matrix evaluation | DSML | Already committed; now maps 1-to-1 onto 3 lectures |
| 2 | LSTM training-practice experiments (optimisers, LR schedules, regularisation, over/underfitting) | DL | Directly covers almost all of DL Units 1–2; needed anyway before the retrain |
| 3 | Regression baselines (Ridge, Random Forest, Gradient Boosting) with MAE/MSE/RMSE/R²/MAPE | DSML | Already committed; uses the taught metrics |
| 4 | LSTM autoencoder in the anomaly comparison | DL (+DSML) | Already committed; on the DSML anomaly slide |
| 5 | RNN vs GRU vs LSTM comparison | DL | Vanishing-gradient material; small extra code |
| 6 | Clustering driving styles (K-Means + elbow, hierarchical, PCA + scree) | DSML | Heavily taught (2 decks + 2 notebooks); currently "undecided" |
| 7 | Classification (tyre compound) with LogReg/KNN/SVM/NB/RF + confusion matrix | DSML | Heavily taught (5 notebooks); currently "undecided" |
| 8 | EDA & statistics section (descriptive stats, outliers, correlation-based feature selection) | DSML | Cheap; covers Units 1–2 of DSML |

Proposed roadmap changes (pending team decision): replace LOF with KNN + DBSCAN in §B; add a
"DL training-practice experiments" item; consider moving clustering+PCA and classification from
"undecided" to "committed"; consider RNN/GRU comparison (previously "not planned").
