# Football Match Visualization (PV251)

Interactive visualization of football match data across multiple European leagues and seasons (2020/21–2024/25).  
Built with **Python, Dash, Plotly, DuckDB**, and **Dash Bootstrap Components**.

The application runs locally and allows users to:
- select a country and competition from a Europe map,
- filter matches using a two-way time slider,
- explore match statistics via interactive charts

---

## Project Overview

- **Frontend / UI**: Dash, Plotly, Dash Bootstrap Components  
- **Backend / Data layer**: DuckDB (built from CSV on first run)  
- **Data size**: main matches CSV ≈ 300 MB (downloaded automatically)

Large data files are **not stored in Git** and are downloaded on demand.

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd pv251_footballviz
```

### 2. Create and activate a virtual environment

Windows:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python scripts/run_dev.py
```
The script download the matches CSV file (if missing), build the DuckDB database (if missing) and starts the Dash app at `http://127.0.0.1:8050/`.

## Troubleshooting
```bash
# delete the database file
rm data/cache/football.duckdb

# run again
python scripts/run_dev.py
```