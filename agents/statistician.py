"""AJAN 2: İstatistikçi & EDA."""
import numpy as np
import pandas as pd


def analyze(df: pd.DataFrame):
    skew = df.select_dtypes("number").skew()
    table = pd.DataFrame({
        "Kolon": df.columns,
        "Tip": df.dtypes.astype(str).values,
        "Eksik %": (df.isna().mean() * 100).round(2).values,
        "Skew": [round(float(skew.get(c, np.nan)), 2) for c in df.columns],
    })
    meta = {"Boyut": f"{df.shape[0]} x {df.shape[1]}", "Mükerrer satır": int(df.duplicated().sum())}

    miss = table.loc[table["Eksik %"].idxmax()]
    f1 = f"Eksik: en yüksek `{miss['Kolon']}` (%{miss['Eksik %']}); mükerrer satır: {meta['Mükerrer satır']}."

    sk = table.dropna(subset=["Skew"])
    if len(sk):
        s = sk.loc[sk["Skew"].abs().idxmax()]
        f2 = f"Çarpıklık: en yüksek `{s['Kolon']}` (skew={s['Skew']}); |skew|>1 olanlar: {int((sk['Skew'].abs() > 1).sum())}."
    else:
        f2 = "Sayısal kolon yok."

    num = df.select_dtypes("number")
    if num.shape[1] > 1:
        c = num.corr().abs().where(~np.eye(num.shape[1], dtype=bool)).stack()
        (a, b), v = c.idxmax(), c.max()
        f3 = f"Korelasyon: en yüksek `{a}` ~ `{b}` (|r|={v:.2f})" + (" -> çoklu doğrusallık riski." if v > 0.9 else ".")
    else:
        f3 = "Korelasyon için yeterli sayısal kolon yok."
    return meta, table, [f1, f2, f3]
