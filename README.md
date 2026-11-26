# MBA Blood Bank Research Prototype

Synthetic data; educational research only. Not for clinical use.

## Commands (Windows PowerShell)
```powershell
cd MBA_Blood_Bank_Prototype
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
If PowerShell blocks activation, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` then `.venv\Scripts\python.exe -m streamlit run app.py`.

## macOS/Linux
```bash
cd MBA_Blood_Bank_Prototype
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open the localhost URL displayed by Streamlit (usually http://localhost:8501). Stop with Ctrl+C.

## App
Four tabs: overview/EDA; chronological holdout comparison (last 12 days) for previous-day, 7-day average, Ridge, Decision Tree, Random Forest; observed order-fulfilment gaps and illustrative stock coverage; raw data and limitations. Models are retrained when the app runs. Results may differ from the earlier Phase 3 workbook due to explicit model parameters and split implementation. Do not present illustrative coverage as a validated clinical risk classification.
