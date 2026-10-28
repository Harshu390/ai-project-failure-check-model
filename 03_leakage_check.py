import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
BASE = Path(__file__).resolve().parent
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
df = pd.read_csv(BASE / "dataset.csv")
real = (df.data_origin=="real").values
ALL = ["data_readiness","leadership_sponsorship","team_ai_maturity","integration_complexity","engineer_buy_in","team_capacity","use_case_fit","poc_before_purchase","scope_narrow"]
CORE = ["data_readiness","leadership_sponsorship","integration_complexity"]   # present for most real rows

# 1) Is missingness itself predictive in the real rows?
r = df[real].copy()
r["n_missing"] = r[ALL].isna().sum(axis=1)
r["src"] = np.where(r.source_quality=="high","vendor_pr","other")
print(r.groupby("failed")["n_missing"].mean().round(2).rename("avg # missing features"))
print("\nFailure rate by source type (real rows):\n", r.groupby("src")["failed"].agg(["count","mean"]))

# 2) Leave-one-real-out using only CORE features
def loro(feats, mk):
    X, y = df[feats], df["failed"]; ps=[]
    for i in np.where(real)[0]:
        tr = np.ones(len(df),bool); tr[i]=False
        m = mk().fit(X[tr], y[tr]); ps.append(m.predict_proba(X.iloc[[i]])[0,1])
    ps=np.array(ps); yt=y[real].values
    return round(accuracy_score(yt,ps>=.5),3), round(roc_auc_score(yt,ps),3)
mk_xgb = lambda: Pipeline([("i",SimpleImputer(strategy="median")),("m",XGBClassifier(n_estimators=150,max_depth=3,learning_rate=.08,eval_metric="logloss",random_state=42))])
mk_lr  = lambda: Pipeline([("i",SimpleImputer(strategy="median")),("m",LogisticRegression(max_iter=1000))])
print("\nLeave-one-real-out (accuracy, AUC):")
print(" XGB all 9 features  :", loro(ALL, mk_xgb))
print(" XGB core 3 features :", loro(CORE, mk_xgb))
print(" LR  all 9 features  :", loro(ALL, mk_lr))
print(" LR  core 3 features :", loro(CORE, mk_lr))
print("\nMajority-class baseline on real rows:", round(9/17,3))
