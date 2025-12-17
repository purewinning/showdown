# DK Showdown Builder (Streamlit)

Upload a projections CSV and generate DraftKings Showdown lineups with:
- Salary + team constraints
- Captain ranking
- Candidate generation + Monte Carlo "first place proxy" scoring
- Exposure + diversity controls for your final set

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
1) Push these files to a GitHub repo  
2) Streamlit Cloud → New app → select repo → main file: `app.py`
