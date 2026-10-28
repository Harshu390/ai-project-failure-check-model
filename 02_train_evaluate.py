import json, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = str(BASE / "charts")
import os; os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv(BASE / "dataset.csv")
FEATS = ["data_readiness","leadership_sponsorship","team_ai_maturity","integration_complexity",
         "engineer_buy_in","team_capacity","use_case_fit","poc_before_purchase","scope_narrow"]
X, y = df[FEATS], df["failed"]
is_real = (df["data_origin"] == "real").values

# ---------- Chart 1: class balance ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ct = df.groupby(["data_origin","outcome"]).size().unstack(fill_value=0)[["scaled","stalled","abandoned"]]
ct.T.plot(kind="bar", ax=ax[0], color=["#2E75B6","#F4B183"], rot=0)
ax[0].set_title("Outcome by data origin"); ax[0].set_ylabel("Cases"); ax[0].set_xlabel("")
bal = df["failed"].map({0:"Scaled (0)",1:"Failed (1)"}).value_counts()
ax[1].bar(bal.index, bal.values, color=["#2E75B6","#C0504D"])
ax[1].set_title("Binary target (all 80 rows)"); ax[1].set_ylabel("Cases")
for i, v in enumerate(bal.values): ax[1].text(i, v+0.5, str(v), ha="center")
plt.tight_layout(); plt.savefig(f"{OUT}/01_class_balance.png", dpi=150); plt.close()

# ---------- Chart 2: correlation heatmap ----------
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[FEATS+["failed"]].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            cbar_kws={"shrink":.8}, annot_kws={"size":8})
plt.title("Correlation heatmap (all rows)"); plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150); plt.close()

# ---------- Models ----------
def make(name):
    imp = SimpleImputer(strategy="median")
    if name == "Logistic Regression":
        return Pipeline([("imp", imp), ("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=1000, C=1.0))])
    if name == "Random Forest":
        return Pipeline([("imp", imp), ("m", RandomForestClassifier(n_estimators=300, max_depth=4,
                         min_samples_leaf=2, random_state=42))])
    return Pipeline([("imp", imp), ("m", XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.08,
                     subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42))])

NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]
metrics = {"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}

# ---------- Evaluation A: repeated stratified 5-fold CV on all 80 rows ----------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
resA = {}
for n in NAMES:
    s = cross_validate(make(n), X, y, cv=cv, scoring=metrics)
    resA[n] = {k: float(np.mean(s[f"test_{k}"])) for k in metrics}
    resA[n].update({k+"_sd": float(np.std(s[f"test_{k}"])) for k in metrics})

# ---------- Evaluation B (honest check): predict the 17 REAL cases ----------
# B1: train on synthetic only -> test on real  (do assumed patterns match real world?)
# B2: leave-one-real-out: train on synthetic + other 16 real, predict the held-out real case
resB1, resB2 = {}, {}
Xr, yr = X[is_real], y[is_real]
Xs, ys = X[~is_real], y[~is_real]
def score(yt, yp, pr):
    return {"accuracy":accuracy_score(yt,yp), "precision":precision_score(yt,yp,zero_division=0),
            "recall":recall_score(yt,yp), "f1":f1_score(yt,yp), "roc_auc":roc_auc_score(yt,pr)}
real_idx = np.where(is_real)[0]
for n in NAMES:
    m = make(n).fit(Xs, ys)
    pr = m.predict_proba(Xr)[:,1]; resB1[n] = score(yr, (pr>=.5).astype(int), pr)
    prs = []
    for i in real_idx:
        train = np.ones(len(df), bool); train[i] = False
        mm = make(n).fit(X[train], y[train]); prs.append(mm.predict_proba(X.iloc[[i]])[:,1][0])
    prs = np.array(prs); resB2[n] = score(yr, (prs>=.5).astype(int), prs)

best = max(NAMES, key=lambda n: resA[n]["f1"] + resA[n]["roc_auc"])
print("BEST (by CV F1+AUC):", best)

# ---------- Chart 3: model comparison ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(NAMES)); w = .2
for j, k in enumerate(["accuracy","precision","recall","f1"]):
    ax.bar(x + (j-1.5)*w, [resA[n][k] for n in NAMES], w, label=k.capitalize())
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylim(0, 1.05)
ax.set_title("Model comparison - repeated 5-fold CV (80 rows)"); ax.legend(ncol=4, loc="lower center")
plt.tight_layout(); plt.savefig(f"{OUT}/03_model_comparison.png", dpi=150); plt.close()

# ---------- Chart 4/5: confusion matrix + ROC (out-of-fold predictions, best model) ----------
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = cross_val_predict(make(best), X, y, cv=skf, method="predict_proba")[:,1]
cm = confusion_matrix(y, (oof>=.5).astype(int))
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Scaled","Failed"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix - {best}"); plt.tight_layout()
plt.savefig(f"{OUT}/04_confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(5.5, 4.5))
for n in NAMES:
    o = cross_val_predict(make(n), X, y, cv=skf, method="predict_proba")[:,1]
    fpr, tpr, _ = roc_curve(y, o); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y,o):.2f})")
plt.plot([0,1],[0,1],"k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (out-of-fold)"); plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/05_roc_curve.png", dpi=150); plt.close()

# ---------- Chart 6: honest check on real rows ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(NAMES)); w = .35
ax.bar(x - w/2, [resB1[n]["accuracy"] for n in NAMES], w, label="Train synthetic -> test real")
ax.bar(x + w/2, [resB2[n]["accuracy"] for n in NAMES], w, label="Leave-one-real-out")
ax.axhline(9/17, color="grey", ls="--", lw=1); ax.text(2.45, 9/17+.01, "majority baseline", ha="right", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy on the 17 real cases")
ax.set_title("Reality check: accuracy on REAL cases only"); ax.legend(loc="lower center")
plt.tight_layout(); plt.savefig(f"{OUT}/06_real_cases_check.png", dpi=150); plt.close()

# ---------- SHAP on best model (fit on all rows) ----------
final = make(best).fit(X, y)
Xi = pd.DataFrame(final.named_steps["imp"].transform(X), columns=FEATS)
mdl = final.named_steps["m"]
if best == "Logistic Regression":
    Xsc = final.named_steps["sc"].transform(Xi)
    expl = shap.LinearExplainer(mdl, Xsc); sv = expl.shap_values(Xsc); Xshow = pd.DataFrame(Xsc, columns=FEATS)
else:
    expl = shap.TreeExplainer(mdl); sv = expl.shap_values(Xi); Xshow = Xi
if isinstance(sv, list): sv = sv[1]
if getattr(sv, "ndim", 2) == 3: sv = sv[:,:,1]
plt.figure(); shap.summary_plot(sv, Xshow, show=False, plot_size=(8, 5))
plt.title(f"SHAP summary - {best} (higher = pushes toward FAILURE)")
plt.tight_layout(); plt.savefig(f"{OUT}/07_shap_summary.png", dpi=150); plt.close()
imp_rank = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False)
plt.figure(figsize=(7, 4.2)); imp_rank[::-1].plot(kind="barh", color="#2E75B6")
plt.title("Mean |SHAP| feature importance"); plt.tight_layout()
plt.savefig(f"{OUT}/08_feature_importance.png", dpi=150); plt.close()

# ---------- Save ----------
pickle.dump({"model": final, "features": FEATS, "best": best,
             "medians": X.median().to_dict()}, open(BASE / "risk_model.pkl", "wb"))
json.dump({"best": best, "A_cv_all80": resA, "B1_train_syn_test_real": resB1,
           "B2_leave_one_real_out": resB2, "shap_rank": imp_rank.round(3).to_dict(),
           "confusion_matrix_oof": cm.tolist()}, open(BASE / "results.json","w"), indent=2)

def tab(d): return pd.DataFrame(d).T[["accuracy","precision","recall","f1","roc_auc"]].round(3)
print("\n=== A) Repeated 5-fold CV, all 80 rows ==="); print(tab(resA))
print("\n=== B1) Train on synthetic only -> test on 17 REAL ==="); print(tab(resB1))
print("\n=== B2) Leave-one-real-out (17 real) ==="); print(tab(resB2))
print("\nSHAP importance:\n", imp_rank.round(3))
print("\nConfusion matrix (OOF):\n", cm)
