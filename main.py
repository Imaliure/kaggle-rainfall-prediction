import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import seaborn as sns


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test.isnull().sum()
test["winddirection"].fillna(test["winddirection"].mode()[0], inplace=True)

y = train["rainfall"]
train_df = train.drop(columns=["id","day","rainfall","mintemp","maxtemp","temparature","winddirection"])
test_df  = test.drop(columns=["id","day","mintemp","maxtemp","temparature", "winddirection"])

for col in train_df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=y, y=train_df[col])  # x = hedef, y = feature
    plt.title(f"{col} vs rainfall")
    plt.show()



corr_matrix = train_df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

RND = 42

train_fe = train_df.copy()
test_fe  = test_df.copy()


# ---------- 3) Baseline CV skorunu alacak helper (OOF üretimi) ----------
def get_oof_preds(model, X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RND)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold = 0
    for train_idx, val_idx in skf.split(X, y):
        fold += 1
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict_proba(X_val)[:,1]
        test_preds += model.predict_proba(X_test)[:,1] / n_splits
        auc_fold = roc_auc_score(y_val, oof[val_idx][val_idx]) if False else roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
        print(f"  fold {fold} AUC: {roc_auc_score(y_val, model.predict_proba(X_val)[:,1]):.4f}")
    return oof, test_preds

# ---------- 4) modelleri hazırla (iyi default parametreler) ----------
count_ones = (y==1).sum()
count_zeros = (y==0).sum()
scale_pos_weight = count_zeros / (count_ones + 1e-6)
print(f"Classes -> 1s: {count_ones}, 0s: {count_zeros}, scale_pos_weight={scale_pos_weight:.3f}")

# LightGBM default strong params
lgb_params = {
    "random_state": RND,
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "class_weight": "balanced"
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# XGBoost default strong params (use scale_pos_weight)
xgb_params = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "use_label_encoder": False,
    "random_state": RND,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1
}
xgb_model = xgb.XGBClassifier(**xgb_params)

# RandomForest strong baseline
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    class_weight="balanced",
    random_state=RND,
    n_jobs=-1
)

# ---------- 5) OOF tahminlerini al (stacking için) ----------
print("\nTraining LGBM with CV...")
oof_lgb, test_lgb = get_oof_preds(lgb_model, train_fe, y, test_fe, n_splits=5)
print("LGBM OOF AUC:", roc_auc_score(y, oof_lgb))

print("\nTraining XGB with CV...")
oof_xgb, test_xgb = get_oof_preds(xgb_model, train_fe, y, test_fe, n_splits=5)
print("XGB OOF AUC:", roc_auc_score(y, oof_xgb))

print("\nTraining RF with CV...")
oof_rf, test_rf = get_oof_preds(rf_model, train_fe, y, test_fe, n_splits=5)
print("RF OOF AUC:", roc_auc_score(y, oof_rf))

# ---------- 6) Meta model için OOF birleştirme ----------
meta_train = np.vstack([oof_lgb, oof_xgb, oof_rf]).T
meta_test  = np.vstack([test_lgb, test_xgb, test_rf]).T

print("Meta train shape:", meta_train.shape)

# Basit meta model: Logistic Regression (regularize)
meta_model = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=RND)
meta_model.fit(meta_train, y)
meta_oof = meta_model.predict_proba(meta_train)[:,1]
meta_auc = roc_auc_score(y, meta_oof)
print(f"\nMeta (stack) OOF AUC: {meta_auc:.4f}")

# ---------- 7) Final: tüm veride base modelleri yeniden eğit ve stacking ile test tahmini ----------
print("\nRefitting base models on full train and predicting test...")
lgb_model.fit(train_fe, y)
xgb_model.fit(train_fe, y)
rf_model.fit(train_fe, y)

test_pred_lgb = lgb_model.predict_proba(test_fe)[:,1]
test_pred_xgb = xgb_model.predict_proba(test_fe)[:,1]
test_pred_rf  = rf_model.predict_proba(test_fe)[:,1]

final_meta_test = np.vstack([test_pred_lgb, test_pred_xgb, test_pred_rf]).T
final_test_proba = meta_model.predict_proba(final_meta_test)[:,1]

# ---------- 8) CV raporu ve submission ----------
print("\nOOF AUC scores (base):",
      roc_auc_score(y, oof_lgb),
      roc_auc_score(y, oof_xgb),
      roc_auc_score(y, oof_rf))
print("Stacked (meta) OOF AUC:", meta_auc)

# ROC curve for stacked oof
fpr, tpr, _ = roc_curve(y, meta_oof)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'Stacked OOF (AUC={meta_auc:.4f})')
plt.plot([0,1],[0,1],"k--")
plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - Stacked OOF")
plt.show()

# save submission (probabilities)
submission = pd.DataFrame({"id": test["id"], "rainfall": final_test_proba})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved. Final stacked test prob head:\n", submission.head())
