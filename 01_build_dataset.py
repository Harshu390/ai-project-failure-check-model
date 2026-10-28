"""
Step 1: Build the dataset.
- 17 REAL coded case studies (values only where the source states or clearly supports them; blank = not stated)
- N SYNTHETIC rows generated from an explicit risk rule (assumptions documented below)
Encoding: 1 = Low / Poor, 2 = Medium, 3 = High / Good.  Binary flags: 1 = Yes, 0 = No.
Target: failed = 1 (stalled in pilot OR abandoned), 0 = scaled to production.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

rng = np.random.default_rng(42)
NA = np.nan

cols = ["case_id", "case", "source", "source_quality", "industry", "use_case",
        "data_readiness", "leadership_sponsorship", "team_ai_maturity",
        "integration_complexity", "engineer_buy_in", "team_capacity",
        "poc_before_purchase", "scope_narrow", "use_case_fit",
        "outcome", "failed"]

real = [
 # id, case, source, quality, industry, use_case, data, lead, maturity, integ, buyin, capacity, poc, scope, fit, outcome
 (1,"Auth0","Datadog case study","high","IT/SaaS","Observability/RCA",3,3,NA,2,NA,NA,NA,NA,NA,"scaled"),
 (2,"USDA Forest Service","ECCO Select / Datadog+AWS","high","Government","Unified observability",2,3,NA,3,NA,NA,NA,NA,NA,"scaled"),
 (3,"Boston Consulting Group","RapDev / Datadog","high","Prof. services","Monitoring consolidation",3,3,NA,2,NA,NA,NA,NA,NA,"scaled"),
 (4,"Singapore Airlines","Splunk case study","high","Aviation","Full-stack observability",3,3,NA,3,NA,NA,NA,NA,NA,"scaled"),
 (5,"Specsavers","Splunk case study","high","Retail/Healthcare","Tool consolidation / incident response",2,3,NA,3,NA,NA,NA,NA,NA,"scaled"),
 (6,"Progressive Insurance","Splunk case study","high","Insurance","Observability + security",3,3,NA,3,NA,NA,NA,NA,NA,"scaled"),
 (7,"UK retail bank","GlobalLogic AIOps case study","high","Banking","AIOps monitoring platform",2,3,NA,3,NA,NA,NA,NA,NA,"scaled"),
 (8,"TalkTalk","Splunk case study","high","Telecom","Network performance monitoring",2,2,NA,2,NA,NA,NA,NA,NA,"scaled"),
 (9,"Acquia","Splunk case study","high","Software/Hosting","Infra monitoring",2,2,NA,1,NA,NA,NA,NA,NA,"scaled"),
 (10,"Melbourne manufacturer","AI pilot-failure article","low","Manufacturing","AI quality control",1,1,NA,3,NA,NA,NA,0,NA,"abandoned"),
 (11,"Automotive OEM","AI pilot-failure article","low","Automotive","Predictive failure detection",2,1,NA,3,NA,NA,NA,NA,NA,"stalled"),
 (12,"Sydney scheduling AI","AI pilot-failure article","low","Prof. services","AI scheduling automation",2,1,NA,2,1,NA,NA,NA,NA,"stalled"),
 (15,"CEO-driven LLM startup","Reddit r/artificial comment (2025)","low","Tech startup","LLM business-rule automation",NA,3,1,NA,1,NA,NA,NA,1,"stalled"),
 (16,"Fortune 100 chatbot bolt-ons","Reddit r/artificial comment (2025)","low","Large enterprise","Chatbot bolt-on apps",NA,3,1,NA,NA,NA,NA,NA,NA,"stalled"),
 (17,"Wall Street firm, 800 apps","Reddit r/artificial comment (2025)","low","Finance","AI-enable custom apps",NA,3,1,NA,NA,NA,NA,0,NA,"stalled"),
 (19,"Unnamed org, 6-wk sandbox POC","Reddit r/sysadmin comment (2022)","low","IT","AIOps ticket reduction",NA,2,NA,NA,3,2,1,NA,NA,"abandoned"),
 (20,"Unnamed MSP","Reddit r/sysadmin comment (2022)","low","IT services","AIOps alert suppression",NA,2,NA,NA,2,2,0,NA,1,"abandoned"),
]
df_real = pd.DataFrame(real, columns=[c for c in cols if c not in ("failed",)])
df_real["failed"] = (df_real["outcome"] != "scaled").astype(int)

# ---------------- Synthetic generation ----------------
# ASSUMPTION (documented in methodology): direction of effects follows the literature
# (readiness, sponsorship, buy-in, capacity, POC, narrow scope and good use-case fit lower failure
# risk; integration complexity raises it). Coefficients are illustrative, NOT estimated from data.
N_SYN = 63
ord_cols = ["data_readiness","leadership_sponsorship","team_ai_maturity","integration_complexity",
            "engineer_buy_in","team_capacity","use_case_fit"]
bin_cols = ["poc_before_purchase","scope_narrow"]

syn = pd.DataFrame({c: rng.integers(1, 4, N_SYN) for c in ord_cols})
for c in bin_cols:
    syn[c] = rng.integers(0, 2, N_SYN)

w = {"data_readiness":-0.9,"leadership_sponsorship":-0.9,"team_ai_maturity":-0.5,
     "integration_complexity":0.5,"engineer_buy_in":-0.8,"team_capacity":-0.5,
     "use_case_fit":-0.6,"poc_before_purchase":-0.7,"scope_narrow":-0.6}
logit = sum(syn[c]*w[c] for c in w) + rng.normal(0, 0.9, N_SYN)
logit = logit - logit.mean()            # centre so roughly half fail
p_fail = 1/(1+np.exp(-logit*1.2))
syn["failed"] = (rng.random(N_SYN) < p_fail).astype(int)
syn["outcome"] = np.where(syn["failed"]==0, "scaled",
                  np.where(rng.random(N_SYN) < 0.7, "stalled", "abandoned"))
syn.insert(0, "case_id", [f"S{i+1:02d}" for i in range(N_SYN)])
syn.insert(1, "case", [f"Synthetic case {i+1}" for i in range(N_SYN)])
syn["source"] = "synthetic"
syn["source_quality"] = "synthetic"
syn["industry"] = rng.choice(["IT","Finance","Retail","Telecom","Manufacturing","Healthcare"], N_SYN)
syn["use_case"] = rng.choice(["Anomaly detection","Incident prediction","Auto-remediation",
                              "Alert correlation","Capacity forecasting"], N_SYN)

df = pd.concat([df_real, syn[df_real.columns]], ignore_index=True)
df.insert(3, "data_origin", np.where(df["source"]=="synthetic", "synthetic", "real"))
df.to_csv(BASE / "dataset.csv", index=False)

print(df.groupby("data_origin")["failed"].agg(["count","sum","mean"]).round(2))
print("\nOutcome counts (real only):\n", df_real["outcome"].value_counts())
print("\nMissing values in REAL rows (share):\n", df_real[ord_cols+bin_cols].isna().mean().round(2))
