"""AJAN 3: Normalizasyon & Özellik Uzmanı. Kullanıcı seçene kadar modellemeye geçilmez."""
import pandas as pd

OPTIONS = {
    "StandardScaler": "Normale yakın, aykırısız veri; lineer modeller",
    "MinMaxScaler": "Sınırlı aralık [0,1]; aykırı az",
    "RobustScaler": "Aykırı değer fazla; medyan/IQR",
    "PowerTransformer": "|skew|>1; çarpık dağılım -> normale yaklaştırır",
    "Yok": "Ağaç tabanlı modeller ölçeklemeye duyarsız",
}


def recommend(df: pd.DataFrame, target: str):
    rows = []
    for c in df.select_dtypes("number").columns.drop(target, errors="ignore"):
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        out = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100) if iqr > 0 else 0.0
        sk = float(s.skew())
        rec = "PowerTransformer" if abs(sk) > 1 else "RobustScaler" if out > 5 else "StandardScaler"
        rows.append({"Kolon": c, "Skew": round(sk, 2), "Aykırı %": round(out, 2), "Öneri": rec})
    matrix = pd.DataFrame(rows)
    default = matrix["Öneri"].mode().iat[0] if len(matrix) else "Yok"
    return matrix, default
