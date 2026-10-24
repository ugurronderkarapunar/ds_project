import pandas as pd
import streamlit as st

from agents import code_reviewer, statistician, feature_engineer, modeler

st.set_page_config(page_title="DS Agents", layout="wide")
st.title("DS Agents: 4 Ajanlı Veri Bilimi Hattı")

ss = st.session_state
ss.setdefault("scaler", None)

with st.sidebar:
    st.header("Veri")
    up = st.file_uploader("CSV / Excel", type=["csv", "xlsx"])
    df = None
    if up:
        df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        target = st.selectbox("Hedef değişken", df.columns)
        test_size = st.slider("Test oranı", 0.1, 0.4, 0.2, 0.05)

t1, t2, t3, t4 = st.tabs(["1 Kod Denetçisi", "2 İstatistikçi", "3 Özellik Uzmanı", "4 Modelleme"])

with t1:
    code = st.text_area("Python kodu yapıştır", height=250)
    f = st.file_uploader("veya .py yükle", type=["py"], key="py")
    if f:
        code = f.read().decode("utf-8")
    if st.button("Denetle") and code:
        for line in code_reviewer.review(code, f.name if f else "kod.py"):
            st.code(line, language=None)

with t2:
    if df is None:
        st.info("Sol menüden veri yükle.")
    else:
        meta, table, findings = statistician.analyze(df)
        st.write(meta)
        st.dataframe(table, use_container_width=True)
        for i, x in enumerate(findings, 1):
            st.markdown(f"**{i}.** {x}")

with t3:
    if df is None:
        st.info("Önce veri yükle.")
    else:
        matrix, default = feature_engineer.recommend(df, target)
        st.dataframe(matrix, use_container_width=True)
        names = list(feature_engineer.OPTIONS)
        choice = st.radio("Strateji seç", names, index=names.index(default),
                          format_func=lambda k: f"{k}: {feature_engineer.OPTIONS[k]}")
        if st.button("Onayla ve Ajan 4'e geç"):
            ss.scaler = choice
        if ss.scaler:
            st.success(f"Seçildi: {ss.scaler}")

with t4:
    if df is None or ss.scaler is None:
        st.warning("Ajan 3'te strateji onaylanmadan modellemeye geçilmez.")
    else:
        if st.button("Modelleri eğit"):
            with st.spinner("Eğitiliyor..."):
                task, metrics = modeler.run(df, target, ss.scaler, test_size)
            st.caption(f"Görev: {task} | Scaler: {ss.scaler} | Split önce, pipeline içinde fit (sızıntı yok)")
            st.dataframe(metrics, use_container_width=True)
        with st.expander("Pipeline kodu"):
            st.code(modeler.source_code(), language="python")
