# Deploy the Mumbai dashboard permanently

The current Arena live preview is temporary. To obtain a stable URL, deploy the Streamlit app to a hosting service.

## Recommended: Streamlit Community Cloud

This is the simplest option for the current project.

### 1. Put the project in GitHub

Create a private or public GitHub repository and upload:

```text
app.py
src/
models/model_bundle.joblib
models/model_metrics.json
data/processed/
requirements-dashboard.txt
README.md
```

Do not upload:

```text
.cdsapirc
.env
any file containing CDS_API_KEY
```

The dashboard only needs the already-trained model and processed files. It does not need the CDS API key at runtime.

### 2. Use the dashboard requirements file

Rename or copy:

```text
requirements-dashboard.txt -> requirements.txt
```

The full `requirements.txt` is intended for training and ERA5 downloads. The smaller dashboard requirements are faster and safer for hosting.

### 3. Deploy

1. Open <https://share.streamlit.io>
2. Sign in with GitHub.
3. Select the repository and branch.
4. Set the main file to `app.py`.
5. Click **Deploy**.

Streamlit will provide a permanent URL such as:

```text
https://your-app-name.streamlit.app
```

The app will sleep when unused on free hosting, but the URL remains stable.

## Important: model updates are separate from dashboard hosting

The deployed dashboard displays the model files already committed to GitHub. It will not automatically download ERA5 or retrain just because the dashboard is open.

To update the model:

```bash
export CDS_API_KEY="YOUR_CDS_PERSONAL_ACCESS_TOKEN"
python scripts/download_era5.py --start-year 2008 --end-year 2018
python scripts/process_era5_timeseries.py
python scripts/prepare_data.py
python -m src.forecast
```

Then commit and push the updated files:

```bash
git add data/processed models
git commit -m "Update ERA5 model outputs"
git push
```

Streamlit Cloud will redeploy automatically.

## Automated updates with GitHub Actions

For a scheduled operational system, add a GitHub Actions workflow that:

1. Runs monthly or weekly.
2. Reads `CDS_API_KEY` from GitHub repository Secrets.
3. Downloads the newest ERA5 data.
4. Adds the newest BMC/Praja disease file.
5. Retrains and validates the model.
6. Commits updated model/forecast artifacts.

Never put the token directly inside a notebook, Python file or GitHub repository.

## Alternatives

### Render

Use a Web Service with:

```text
Build command: pip install -r requirements-dashboard.txt
Start command: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Render is better when you need a continuously running service or a custom domain.

### Hugging Face Spaces

Create a Streamlit Space, upload the project files and set the SDK to Streamlit. This is useful for public research demonstrations.

### Private production deployment

For BMC or institutional use, deploy on a managed VM, Azure App Service, AWS, GCP or an institutional server. Put the model behind authentication and add a database for incoming monthly surveillance data.

## Production checklist

Before sharing the URL publicly:

- Keep the CDS token out of GitHub and the dashboard.
- Restrict the dashboard if ward health data are sensitive.
- Show the data coverage and forecast limitation on the page.
- Add authentication for operational use.
- Log each model version, data vintage and forecast date.
- Add a scheduled prospective backtest using newer observed cases.
- Do not use the dashboard alone to declare an outbreak.
