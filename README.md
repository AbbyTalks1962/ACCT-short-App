# Equity Screening Lab

A Streamlit app implementing a three-stage sequential stock-screening workflow with adjustable elimination percentages, industry demeaning, rank recalculation, a final portfolio view, and CSV upload/download.

## Important data note

`sample_700_companies.csv` contains 700 **synthetic demo companies and synthetic ratios**. These are for testing the app only, not actual public-company financial data. Replace them by uploading a verified CSV with the required columns.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push this folder to a GitHub repository, then deploy `app.py` using Streamlit Community Cloud. See https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Required input columns

`Company, Ticker, Sector, Industry, STA, SNOA, Z_Score, MS, MG, GPA, BTM, CFOA`

- STA and SNOA: low is better; optionally demeaned by industry.
- Z_Score: high is better.
- MS: average growth margin / standard deviation of growth margin (calculate using the exact course-defined period).
- MG: long-term geometric growth in margin (calculate using the exact course formula).
- MM: max(percentile(MS), percentile(MG)); high is treated as better.
- GPA, BTM, CFOA: high is treated as better in this version. Confirm BTM direction against your course methodology.

The app recalculates ranks within the survivor population between stages. Stage elimination is based on the highest cumulative rank score (weakest), because rank 1 is best.


## Seven-year MS and MG calculations

For automatic calculation, the CSV includes eight annual margin observations (`Margin_Tminus7` through `Margin_T`), which yield seven year-over-year growth rates. MS is the arithmetic mean of those seven growth rates divided by their sample standard deviation. MG is the annualized geometric growth rate from the first to the last margin: `(ending margin / starting margin)^(1/7) - 1`. The included margin histories are synthetic demo values, not actual company data.
