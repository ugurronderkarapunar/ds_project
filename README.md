# DS Agents

4 ajanlı Streamlit uygulaması: Kod Denetçisi → İstatistikçi → Özellik Uzmanı → Modelleme.

## Yerel çalıştırma
```
pip install -r requirements.txt
streamlit run app.py
```

## GitHub + Streamlit Cloud
1. `git init && git add . && git commit -m "init"`
2. GitHub'da repo aç, `git remote add origin <url> && git push -u origin main`
3. share.streamlit.io -> New app -> repo seç -> Main file: `app.py` -> Deploy
