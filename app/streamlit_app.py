from pathlib import Path
import pandas as pd
import streamlit as st

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "processed" / "bls_timeseries.csv"

# Friendly labels (you can expand later)
SERIES_NAMES = {
    "LNS14000000": "Unemployment Rate (SA)",
    "CES0000000001": "Total Nonfarm Employment (SA)",
    "CES0500000002": "Avg Weekly Hours - Total Private (SA)",
    "CES0500000003": "Avg Hourly Earnings - Total Private (SA)",
    "PRS85006092": "Output Per Hour - Nonfarm Business Productivity",
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df

def compute_metrics(pivot: pd.DataFrame):
    """Return a DataFrame of latest, MoM %, YoY % for each series."""
    out = []
    for sid in pivot.columns:
        s = pivot[sid].dropna().sort_index()
        if len(s) < 2:
            continue
        latest_date = s.index.max()
        latest_val = float(s.loc[latest_date])

        # Month-over-month
        prev_val = float(s.iloc[-2])
        mom = (latest_val - prev_val) / prev_val * 100 if prev_val != 0 else None

        # Year-over-year (same month last year, if present)
        prior_year_date = latest_date - pd.DateOffset(years=1)
        if prior_year_date in s.index:
            yoy_val = float(s.loc[prior_year_date])
            yoy = (latest_val - yoy_val) / yoy_val * 100 if yoy_val != 0 else None
        else:
            yoy = None

        out.append({
            "series_id": sid,
            "latest_date": latest_date,
            "latest_value": latest_val,
            "mom_pct": mom,
            "yoy_pct": yoy
        })
    return pd.DataFrame(out)

def arrow(x):
    if x is None or pd.isna(x):
        return "→"
    if x > 0:
        return "▲"
    if x < 0:
        return "▼"
    return "→"

def main():
    st.set_page_config(page_title="BLS Labor Dashboard", layout="wide")
    st.title("BLS Labor Market Dashboard")
    st.caption("This dashboard reads data stored in the GitHub repo. A GitHub Action updates it monthly (no API calls on page load).")

    # Load and pivot
    df = load_data()
    pivot = df.pivot(index="date", columns="series_id", values="value").sort_index()

    # --- Section 1: Headline metrics ---
    st.subheader("Headline Indicators")
    metrics = compute_metrics(pivot)

    if metrics.empty:
        st.warning("No data found yet. Run the fetch script to create bls_timeseries.csv.")
        return

    cols = st.columns(min(5, len(metrics)))
    for i, row in metrics.head(5).iterrows():
        sid = row["series_id"]
        name = SERIES_NAMES.get(sid, sid)
        mom = row["mom_pct"]
        yoy = row["yoy_pct"]
        latest = row["latest_value"]
        d = row["latest_date"].date()

        with cols[i % len(cols)]:
            st.metric(
                label=name,
                value=f"{latest:,.2f}",
                delta=f"{arrow(mom)} {mom:.2f}% MoM" if mom is not None else "→ MoM n/a"
            )
            st.caption(f"{arrow(yoy)} {yoy:.2f}% YoY • Latest: {d}" if yoy is not None else f"YoY n/a • Latest: {d}")

    st.divider()

    # --- Section 2: Time-series explorer ---
    st.subheader("Time-Series Explorer")

    series_options = list(pivot.columns)
    left, right = st.columns(2)
    with left:
        primary = st.selectbox(
            "Primary series",
            series_options,
            index=series_options.index("LNS14000000") if "LNS14000000" in series_options else 0,
            format_func=lambda x: SERIES_NAMES.get(x, x),
        )
    with right:
        compare = st.selectbox(
            "Comparison series (optional)",
            ["(none)"] + series_options,
            index=0,
            format_func=lambda x: SERIES_NAMES.get(x, x) if x != "(none)" else "(none)",
        )

    # Convert pandas timestamps to plain Python datetimes for Streamlit slider
    min_date = pivot.index.min().to_pydatetime()
    max_date = pivot.index.max().to_pydatetime()
    default_start = (pd.Timestamp(max_date) - pd.DateOffset(months=24))
    default_start = max(pd.Timestamp(min_date), default_start).to_pydatetime()

    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM",
    )

    # Convert slider output back to pandas timestamps for filtering
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])

    mask = (pivot.index >= start_ts) & (pivot.index <= end_ts)
    ts = pivot.loc[mask]

    st.line_chart(ts[[primary]].rename(columns={primary: SERIES_NAMES.get(primary, primary)}))

    if compare != "(none)":
        st.line_chart(ts[[primary, compare]].rename(columns={
            primary: SERIES_NAMES.get(primary, primary),
            compare: SERIES_NAMES.get(compare, compare),
        }))

    st.divider()

    # --- Section 3: Comparative scatter ---
    st.subheader("Comparative Analysis (Scatter)")

    x = st.selectbox("X-axis", series_options, format_func=lambda s: SERIES_NAMES.get(s, s))
    y = st.selectbox("Y-axis", series_options, index=1 if len(series_options) > 1 else 0,
                     format_func=lambda s: SERIES_NAMES.get(s, s))

    scatter = pivot[[x, y]].dropna()
    scatter = scatter.rename(columns={x: SERIES_NAMES.get(x, x), y: SERIES_NAMES.get(y, y)})

    st.scatter_chart(scatter)

    st.divider()

    # --- Section 4: Download ---
    st.subheader("Download Data")

    chosen = st.multiselect(
        "Choose series to download",
        series_options,
        default=series_options,
        format_func=lambda s: SERIES_NAMES.get(s, s)
    )

    export = pivot[chosen].reset_index()
    st.download_button(
        "Download selected data as CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="bls_selected_series.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()