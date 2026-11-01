# Data Science Internship Projects

Six tasks covering data science fundamentals, a hands-on lab, a real-world dataset analysis, tool mastery, business analytics reporting and a final capstone.

## Structure
- `scripts/` one runnable Python script per task (each writes its charts and a results JSON to `charts/`)
- `data/` public datasets from the seaborn-data repository (titanic, tips, taxis, diamonds) plus the cleaned Titanic file
- `charts/` generated figures and result files used in the reports

## Run it
```bash
python -m venv ds-lab && source ds-lab/bin/activate
pip install -r requirements.txt
cd scripts
python task1_fundamentals.py   # ... through task6_capstone_diamonds.py
```

| Task | Script | Topic |
|---|---|---|
| 1 | task1_fundamentals.py | Python basics, data types, statistics |
| 2 | task2_lab_tips.py | Environment setup and practical exercises (tips dataset) |
| 3 | task3_titanic_eda.py | Cleaning, EDA and visualisation (Titanic) |
| 4 | task4_tool_mastery.py | NumPy, pandas, SQL, scikit-learn pipelines |
| 5 | task5_taxi_analytics.py | Business analytics (NYC taxi trips) |
| 6 | task6_capstone_diamonds.py | Diamond price prediction (regression) |

Data source: https://github.com/mwaskom/seaborn-data
