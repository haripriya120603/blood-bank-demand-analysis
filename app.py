import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Blood Bank Demand & Inventory Risk", layout="wide")
st.title("Blood Bank Demand & Inventory Risk — Research Prototype")
st.caption("Synthetic dataset • Educational proof of concept • Not for clinical decisions")
@st.cache_data
def load():
    with open(Path(__file__).parent/"blood_bank_data.json",encoding="utf-8") as f: raw=json.load(f)
    orders=pd.DataFrame(raw["blood_orders"])
    orders["date"]=pd.to_datetime(orders["ordered_at"]).dt.normalize()
    stock=pd.DataFrame(raw["daily_stock"])
    stock["stock_date"]=pd.to_datetime(stock["stock_date"])
    return orders,stock,raw
orders,stock,raw=load()
types=["O+","A+","B+","AB+","O-","A-","B-","AB-"]
stock_cols={"O+":"o_pos_units","A+":"a_pos_units","B+":"b_pos_units","AB+":"ab_pos_units",
            "O-":"o_neg_units","A-":"a_neg_units","B-":"b_neg_units","AB-":"ab_neg_units"}
tab1,tab2,tab3,tab4=st.tabs(["Overview & EDA","Forecast evaluation","Inventory risk indicators","Data & limitations"])
with tab1:
    a,b,c,d=st.columns(4)
    requested=int(orders.units_requested.sum()); issued=int(orders.units_issued.sum())
    a.metric("Orders",len(orders));b.metric("Units requested",requested)
    c.metric("Units issued",issued);d.metric("Unfulfilled requested units",requested-issued)
    bytype=orders.groupby("blood_type",as_index=False)[["units_requested","units_issued"]].sum()
    st.plotly_chart(px.bar(bytype,x="blood_type",y=["units_requested","units_issued"],barmode="group",
                           title="Requested and issued units by blood type"),use_container_width=True)
    daily=orders.groupby("date",as_index=False)[["units_requested","units_issued"]].sum()
    all_dates=pd.date_range(daily.date.min(),daily.date.max(),freq="D")
    daily=daily.set_index("date").reindex(all_dates,fill_value=0).rename_axis("date").reset_index()
    st.plotly_chart(px.line(daily,x="date",y=["units_requested","units_issued"],title="Daily order activity"),
                    use_container_width=True)
with tab2:
    st.write("Forecasts **total daily requested units**. The final 12 days are held out chronologically. Features use only preceding days.")
    series=daily.set_index("date")["units_requested"].astype(float)
    def features(s):
        rows=[]
        for i in range(7,len(s)):
            history=s.iloc[:i]
            rows.append({"date":s.index[i],"lag1":float(history.iloc[-1]),
                         "lag7":float(history.iloc[-7]),"mean7":float(history.iloc[-7:].mean()),
                         "weekday":int(s.index[i].dayofweek),"target":float(s.iloc[i])})
        return pd.DataFrame(rows).set_index("date")
    feat=features(series)
    X=feat.drop(columns="target");y=feat.target
    split=len(feat)-12
    Xtr,Xte=X.iloc[:split],X.iloc[split:];ytr,yte=y.iloc[:split],y.iloc[split:]
    models={"Ridge Regression":Ridge(alpha=10.0),
            "Decision Tree":DecisionTreeRegressor(max_depth=3,min_samples_leaf=3,random_state=42),
            "Random Forest":RandomForestRegressor(n_estimators=200,max_depth=4,min_samples_leaf=2,random_state=42)}
    predictions={"Previous-day baseline":Xte.lag1.values,"7-day average baseline":Xte.mean7.values}
    for name,model in models.items():
        model.fit(Xtr,ytr)
        predictions[name]=np.maximum(0,model.predict(Xte))
    metrics=pd.DataFrame([{"Model":name,"MAE":mean_absolute_error(yte,p),
                           "RMSE":np.sqrt(mean_squared_error(yte,p))}
                          for name,p in predictions.items()]).sort_values("MAE")
    st.dataframe(metrics.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}"}),hide_index=True,use_container_width=True)
    selected=st.selectbox("Show held-out predictions",list(predictions))
    comparison=pd.DataFrame({"date":yte.index,"Actual":yte.values,"Predicted":predictions[selected]})
    st.plotly_chart(px.line(comparison,x="date",y=["Actual","Predicted"],markers=True,
                            title="Chronological holdout: actual vs predicted"),use_container_width=True)
    st.download_button("Download evaluation CSV",comparison.to_csv(index=False).encode(),
                       file_name="forecast_evaluation.csv",mime="text/csv")
    st.info("This is a held-out evaluation, not a validated long-term forecast. Only 60 daily observations are available.")
with tab3:
    st.write("Two separate descriptive indicators are shown. They are **not clinical shortage predictions**.")
    summary=orders.groupby("blood_type",as_index=False)[["units_requested","units_issued"]].sum()
    summary["unfulfilled_units"]=summary.units_requested-summary.units_issued
    summary["unfulfilled_pct"]=np.where(summary.units_requested>0,100*summary.unfulfilled_units/summary.units_requested,0)
    st.subheader("Observed order-fulfilment gap")
    st.dataframe(summary.sort_values("unfulfilled_pct",ascending=False),hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(summary,x="blood_type",y="unfulfilled_units",title="Unfulfilled requested units"),
                    use_container_width=True)
    st.subheader("Illustrative stock coverage")
    st.caption("Latest recorded stock summed across centres divided by historical average daily requests. Stock snapshots are not component-specific and cannot be reconciled to orders; this is only an illustrative ratio.")
    latest=stock.stock_date.max()
    last=stock[stock.stock_date.eq(latest)]
    days=(orders.date.max()-orders.date.min()).days+1
    rows=[]
    for bt in types:
        available=int(last[stock_cols[bt]].sum())
        demand=int(orders.loc[orders.blood_type.eq(bt),"units_requested"].sum())
        avg=demand/days
        coverage=available/avg if avg else np.nan
        rows.append({"Blood type":bt,"Last-date stock units":available,"Mean requested units/day":round(avg,2),
                     "Illustrative stock coverage (days)":round(coverage,1) if np.isfinite(coverage) else None})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.warning("Do not infer that high coverage means adequate real-world supply: component compatibility, expiry, transfers, timing, and stock-order reconciliation are unavailable.")
with tab4:
    st.write("Original source tables and sample records")
    table=st.selectbox("Table",list(raw))
    st.dataframe(pd.DataFrame(raw[table]),use_container_width=True)
    st.markdown("""**Research limitations:** Synthetic/non-Tamil-Nadu records; only 60 days; no seasonal validation;
    stock snapshots are not reconciled with donations and orders; stock is not component-specific;
    no expiry ledger; issued-vs-requested gaps are not verified clinical shortages. No future or
    same-day target information is used in forecasting features. All displayed risk measures are illustrative.""")
