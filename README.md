# Pangaea nitrate δ15N explorer (Streamlit)

## Files
- `app.py` : Streamlit application
- `requirements.txt` : Python dependencies
- `Pangaea.csv` : your data file (place in same folder, or edit path in the sidebar)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push these files to a GitHub repo
2. Create a new Streamlit app pointing to `app.py`
3. Put `Pangaea.csv` in the repo OR host it somewhere and change the path logic
