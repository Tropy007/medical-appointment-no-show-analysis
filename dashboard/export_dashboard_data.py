"""Export the small, pre-aggregated tables behind the Tableau and Power BI dashboards.

Run from the repository root:  python dashboard/export_dashboard_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(name); print(df.to_string(index=False)); print()
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
df = pd.read_csv(ROOT / "data" / "KaggleV2-May-2016.csv")
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"]).dt.tz_localize(None)
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.tz_localize(None)
df["LeadDays"] = (df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()).dt.days
c = df[(df["Age"] >= 0) & (df["LeadDays"] >= 0)].copy()
c["Missed"] = (c["No-show"] == "Yes").astype(int)
c["Weekday"] = c["AppointmentDay"].dt.day_name()
c["LeadBand"] = pd.cut(c["LeadDays"], [-1, 0, 7, 30, 1000], labels=["1. Same day", "2. 1-7 days", "3. 8-30 days", "4. 31+ days"])
c["AgeGroup"] = pd.cut(c["Age"], [-1, 12, 18, 35, 50, 65, 120], labels=["1. 0-12", "2. 13-18", "3. 19-35", "4. 36-50", "5. 51-65", "6. 66+"])
save(c.groupby("LeadBand", observed=True)["Missed"].mean().round(4).reset_index().rename(columns={"LeadBand": "Booking lead time", "Missed": "No-show rate"}), "noshow_by_lead.csv")
save(c.groupby("AgeGroup", observed=True)["Missed"].mean().round(4).reset_index().rename(columns={"AgeGroup": "Age group", "Missed": "No-show rate"}), "noshow_by_age.csv")
s = c[c["LeadDays"] >= 3].copy()
s["LeadBand"] = s["LeadBand"].astype(str).str.replace("1-7", "3-7")
s["SMS"] = s["SMS_received"].map({0: "No SMS", 1: "SMS received"})
save(s.groupby(["LeadBand", "SMS"])["Missed"].mean().round(4).reset_index().rename(columns={"LeadBand": "Booking lead time", "Missed": "No-show rate"}), "noshow_sms.csv")
feats = ["Age", "Gender", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received", "LeadDays", "Weekday"]
X = c[feats].copy(); X["ScheduledHour"] = c["ScheduledDay"].dt.hour; y = c["Missed"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cat = ["Gender", "Weekday"]; num = [k for k in X.columns if k not in cat]
prep = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
gb = make_pipeline(prep, HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced")).fit(X_tr, y_tr)
p = gb.predict_proba(X_te)[:, 1]
dec = pd.qcut(p, 10, labels=[f"{i:02d}" for i in range(1, 11)])
r = pd.Series(y_te.values).groupby(np.asarray(dec)).mean().round(4).reset_index()
r.columns = ["Risk decile (10 = riskiest)", "No-show rate"]
save(r, "noshow_risk_deciles.csv")
top20 = y_te[p >= np.quantile(p, 0.8)].mean()
save(pd.DataFrame({"metric": ["Appointments", "No-show rate", "Same-day no-show", "31+ days no-show", "Model ROC-AUC", "Missed visits caught"],
                   "value": [f"{len(c):,}", f"{y.mean():.1%}", f"{c[c.LeadDays == 0].Missed.mean():.1%}", f"{c[c.LeadDays > 30].Missed.mean():.1%}",
                             f"{roc_auc_score(y_te, p):.2f}", f"{recall_score(y_te, p >= 0.5):.0%}"]}), "noshow_kpis.csv")
print(f"top 20% rate {top20:.3f}")
