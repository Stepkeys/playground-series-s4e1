# Playground Series S4E1

## Project layout

- `data/raw/`: original competition CSV files
- `data/processed/`: generated or cleaned datasets; see its README
- `notebooks/`: exploratory analysis and modeling notebooks
- `src/`: reusable preprocessing and modeling code, including the shared CSV loader
- `models/`: saved model artifacts
- `outputs/`: predictions, plots, and generated files; see its README
- `reports/`: written analysis and project notes, with an analysis template

## Environment setup

The project uses the existing `.venv` virtual environment. In PowerShell, activate it and install the project dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

VS Code is configured to use `.venv` as the Python interpreter. Open `notebooks/notebook.ipynb` and select the `.venv` kernel if prompted.

## Run the churn prediction app

After installing the project dependencies, start the Streamlit app from the project root:

```powershell
streamlit run app.py
```

The app loads the trained model from `models/best_model.pkl`.
