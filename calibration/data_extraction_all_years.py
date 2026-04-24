"""
Phase 0: extract annual C, P, and alpha matrices from data (1988–2024).

Phi_space are fixed at YEAR_REF (2000); beta_C uses the full series.
Intrinsic growth rates r_C and r_P are estimated by OLS on log-exports over the full series.

Outputs (extracted_data/):
  annual/alpha_{year}.npy  (SC × SP)  row-normalised RCA per year
  annual/C_{year}.npy      (SC,)      normalised country export totals
  annual/P_{year}.npy      (SP,)      normalised world product exports
  phi_space.npy            (SP × SP)  proximity matrix (from YEAR_REF)
  beta_C_ref.npy           (SC × SP)  capability weights in [0,1]: frequency x intensity of exports across full time series (0 if never exported)
  alpha_init.npy           (SC × SP)  initial alpha (YEAR_START)
  P_init.npy               (SP,)      initial P (YEAR_START)
  C_init.npy               (SC,)      initial C (YEAR_START)
  r_P.npy                  (SP,)      OLS trend growth rate per product
  r_C.npy                  (SC,)      OLS trend growth rate per country
  products_index.csv                  product position → hs code + name
  countries_index.csv                 country position → iso3 + name
  coverage.csv                        data coverage fraction per year
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from ecomplexity import proximity

try:
    from calibration_config import YEAR_START, YEAR_END, YEAR_REF, RAW_DATA_DIR, EXTRACTED_DIR
except ModuleNotFoundError as exc:
    if exc.name != "calibration_config":
        raise
    from .calibration_config import YEAR_START, YEAR_END, YEAR_REF, RAW_DATA_DIR, EXTRACTED_DIR

# Settings
# YEAR_REF follows Hidalgo et al. (2007) for defining the product space and network alignment.
IN_RANKINGS_ONLY = True # Whether to include only countries in the annual rankings (G20) or all available countries in trade data.
TOP_N_PRODUCTS = 100

RAW_DIR = RAW_DATA_DIR
OUT_DIR = EXTRACTED_DIR
ANN_DIR = os.path.join(OUT_DIR, "annual")
os.makedirs(ANN_DIR, exist_ok=True)

trade_cols = {
    "time": "year",
    "loc" : "location_code",
    "prod": "hs_product_code",
    "val" : "export_value",
}

# Load files
print("Loading files...")
trade = pd.read_csv(os.path.join(RAW_DIR, "hs92_country_product_year_4.csv"), low_memory=False)
products_lookup = pd.read_csv(os.path.join(RAW_DIR, "product_hs92.csv"), low_memory=False)
countries_lookup = pd.read_csv(os.path.join(RAW_DIR, "location_country.csv"), low_memory=False)


# Filter files

products_lookup = products_lookup[products_lookup["product_level"] == 4].copy()

if IN_RANKINGS_ONLY:
    countries_lookup = countries_lookup[countries_lookup["in_rankings"] == True].copy()

# G20 countries
G20_ISO3 = {
    "ARG", "AUS", "BRA", "CAN", "CHN", "FRA", "DEU", "IND", "IDN",
    "ITA", "JPN", "MEX", "RUS", "SAU", "ZAF", "KOR", "TUR", "GBR", "USA",
}
countries_lookup = countries_lookup[countries_lookup["country_iso3_code"].isin(G20_ISO3)].copy()

valid_countries = countries_lookup["country_iso3_code"].tolist()
valid_products  = products_lookup["product_hs92_code"].tolist()

print(f"Products in lookup (4-digit): {len(valid_products)}")
print(f"Countries in lookup (G20):    {len(valid_countries)}")



# Filter trade data
def filter_year(df, year):
    d = df[df["year"] == year].copy()
    d = d[d["country_iso3_code"].isin(valid_countries)]
    d = d[d["product_hs92_code"].isin(valid_products)]
    d = d[d["export_value"] > 0].dropna(subset=["export_value"])
    d = d.rename(columns={
        "country_iso3_code": "location_code",
        "product_hs92_code": "hs_product_code",
    })
    return d



# Define network from YEAR_REF
print(f"Defining network from reference year {YEAR_REF}...")
df_ref = filter_year(trade, YEAR_REF)

# Keep only top N products by export value in YEAR_REF
top_products = (df_ref.groupby("hs_product_code")["export_value"]
                      .sum().nlargest(TOP_N_PRODUCTS).index.tolist())
df_ref = df_ref[df_ref["hs_product_code"].isin(top_products)]

# Reshape to get RCA matrix for YEAR_REF (countries × products)
rca_ref = df_ref.pivot_table(
    index="location_code", columns="hs_product_code",
    values="export_rca", aggfunc="first"
).fillna(0)

COUNTRIES = rca_ref.index.tolist()
PRODUCTS = rca_ref.columns.tolist()
SC, SP = len(COUNTRIES), len(PRODUCTS)
print(f"Countries (SC): {SC},  Products (SP): {SP}")



# Save index files (position → code + name) for countries and products, aligned to the fixed network
products_df = (
    pd.DataFrame({"position": range(SP), "hs_product_code": PRODUCTS})
    .merge(
        products_lookup[["product_hs92_code", "product_name_short", "product_name"]]
        .rename(columns={"product_hs92_code": "hs_product_code"}),
        on="hs_product_code", how="left"
    )
)
products_df.to_csv(os.path.join(OUT_DIR, "products_index.csv"), index=False)

countries_df = (
    pd.DataFrame({"position": range(SC), "location_code": COUNTRIES})
    .merge(
        countries_lookup[["country_iso3_code", "country_name", "country_name_short"]]
        .rename(columns={"country_iso3_code": "location_code"}),
        on="location_code", how="left"
    )
)
countries_df.to_csv(os.path.join(OUT_DIR, "countries_index.csv"), index=False)



# phi_space from YEAR_REF
print("Computing proximity matrix...")
prox_df = proximity(df_ref, trade_cols)
phi_space = (
    prox_df.pivot(index="hs_product_code_1", columns="hs_product_code_2", values="proximity")
    .reindex(index=PRODUCTS, columns=PRODUCTS).fillna(0.0).values.astype(float)
)
np.fill_diagonal(phi_space, 0.0)
np.save(os.path.join(OUT_DIR, "phi_space.npy"), phi_space)
print(f"phi_space: min={phi_space.min():.3f}  max={phi_space.max():.3f}")



# Extract matrices for one year using fixed network
def extract_year(year):
    '''
    Extract alpha, C, P for a given year aligned to the fixed network.
    Missing entries filled with 0.
    '''
    df = filter_year(trade, year)
    df = df[df["hs_product_code"].isin(PRODUCTS)]

    rca_t = (
        df.pivot_table(index="location_code", columns="hs_product_code",
                       values="export_rca", aggfunc="first")
        .reindex(index=COUNTRIES, columns=PRODUCTS).fillna(0.0)
    )
    rca_vals = rca_t.values.copy()

    # Row-normalised RCA (alpha)
    row_sums = rca_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    alpha_t = rca_vals / row_sums

    # World exports per product
    world_exp = (df.groupby("hs_product_code")["export_value"]
                   .sum().reindex(PRODUCTS).fillna(0.0).values.astype(float))
    mean_P    = world_exp[world_exp > 0].mean() if (world_exp > 0).any() else 1.0
    P_t = world_exp / mean_P
    P_t = np.where(P_t == 0, 0.01, P_t) # Avoid zeros

    # Total exports per country
    total_exp = (df.groupby("location_code")["export_value"]
                   .sum().reindex(COUNTRIES).fillna(0.0).values.astype(float))
    mean_C = total_exp[total_exp > 0].mean() if (total_exp > 0).any() else 1.0
    C_t = total_exp / mean_C
    C_t = np.where(C_t == 0, 0.01, C_t)

    coverage = len(df) / (SC * SP) # Fraction of the full country-product space that has data for this year
    return alpha_t, C_t, P_t, coverage, rca_vals



# Extract and save all years
print(f"\nExtracting annual data {YEAR_START}–{YEAR_END}...")
coverage_rows = []
P_series = {}
C_series = {}
rca_series = {}

for year in range(YEAR_START, YEAR_END + 1):
    alpha_t, C_t, P_t, cov, rca_vals = extract_year(year)
    np.save(os.path.join(ANN_DIR, f"alpha_{year}.npy"), alpha_t)
    np.save(os.path.join(ANN_DIR, f"C_{year}.npy"), C_t)
    np.save(os.path.join(ANN_DIR, f"P_{year}.npy"), P_t)
    P_series[year] = P_t
    C_series[year] = C_t
    rca_series[year] = rca_vals
    coverage_rows.append({"year": year, "coverage": cov})
    print(f"  {year}: coverage={cov:.2%}")

pd.DataFrame(coverage_rows).to_csv(os.path.join(OUT_DIR, "coverage.csv"), index=False)



# beta_C from full time series (1988–2024)
# Entry (j,i) is > 0 iff country j ever exported product i (RCA > 0).
# Weight = frequency (fraction of years with any exports) x intensity (mean RCA when active, normalised).
print("\nComputing beta_C_ref from full time series...")
rca_stack = np.stack([rca_series[y] for y in range(YEAR_START, YEAR_END + 1)], axis=0)  # (n_years, SC, SP)

ever_exported = (rca_stack > 0).any(axis=0) # (SC, SP) bool
frequency = (rca_stack > 0).mean(axis=0) # Fraction of years with any export

rca_when_active = np.where(rca_stack > 0, rca_stack, np.nan)
mean_rca_active = np.nanmean(rca_when_active, axis=0) # nan where never exported
mean_rca_active = np.where(ever_exported, mean_rca_active, 0.0)

# Normalise intensity: cap at 95th percentile of active entries to limit outlier influence
rca_cap  = np.percentile(mean_rca_active[ever_exported], 95) if ever_exported.any() else 1.0
intensity = np.clip(mean_rca_active / rca_cap, 0.0, 1.0)

beta_C_ref = np.where(ever_exported, frequency * intensity, 0.0)
np.save(os.path.join(OUT_DIR, "beta_C_ref.npy"), beta_C_ref)
print(f"beta_C_ref: zero entries={( ~ever_exported).mean():.2%}  mean={beta_C_ref[ever_exported].mean():.3f}  max={beta_C_ref.max():.3f}")

# Save initial conditions
np.save(os.path.join(OUT_DIR, "alpha_init.npy"), np.load(os.path.join(ANN_DIR, f"alpha_{YEAR_START}.npy")))
np.save(os.path.join(OUT_DIR, "P_init.npy"), np.load(os.path.join(ANN_DIR, f"P_{YEAR_START}.npy")))
np.save(os.path.join(OUT_DIR, "C_init.npy"), np.load(os.path.join(ANN_DIR, f"C_{YEAR_START}.npy")))


# OLS trend growth rates
print("\nEstimating OLS trend growth rates...")
years_arr = np.array(sorted(P_series.keys()), dtype=float)
t_arr = years_arr - years_arr[0] # t=0 at YEAR_START
P_matrix = np.stack([P_series[int(y)] for y in years_arr], axis=0) # (n_years, SP)
C_matrix = np.stack([C_series[int(y)] for y in years_arr], axis=0) # (n_years, SC)

def ols_growth_rate(matrix, t):
    """
    Estimate OLS slope of log-values over time for each column in the matrix.
    """
    n_series = matrix.shape[1]
    slopes = np.full(n_series, np.nan)
    for i in range(n_series):
        y = matrix[:, i]
        pos_mask = y > 0
        if pos_mask.sum() < 3: # Need at least 3 points for OLS
            continue
        slope, *_ = stats.linregress(t[pos_mask], np.log(y[pos_mask])) # Slope of OLS line for C and P is the log growth rate per year
        slopes[i] = slope
    return slopes

r_P_raw = ols_growth_rate(P_matrix, t_arr)
r_C_raw = ols_growth_rate(C_matrix, t_arr)

# Clip at 1st/99th percentile
def clip_rates(rates, name):
    finite = rates[np.isfinite(rates)]
    p1, p99 = np.percentile(finite, [1, 99])
    n_clipped = ((rates < p1) | (rates > p99)).sum()
    if n_clipped:
        print(f"  {name}: clipping bounds=[{p1:.4f}, {p99:.4f}],  n_clipped={n_clipped}")
    clipped = np.clip(rates, p1, p99)
    clipped = np.where(np.isfinite(clipped), clipped, np.nanmedian(clipped)) # Replace any remaining NaN
    return clipped

r_P = clip_rates(r_P_raw, "r_P")
r_C = clip_rates(r_C_raw, "r_C")

np.save(os.path.join(OUT_DIR, "r_P.npy"), r_P)
np.save(os.path.join(OUT_DIR, "r_C.npy"), r_C)



# Summary
print(f"\nSummary")
print(f"phi_space  : {phi_space.shape}  | min={phi_space.min():.3f}  max={phi_space.max():.3f}")
print(f"beta_C_ref : {beta_C_ref.shape}  | sparsity={1 - beta_C_ref.mean():.2%}")
print(f"r_P        : {r_P.shape}  | mean={r_P.mean():.4f}  std={r_P.std():.4f}")
print(f"r_C        : {r_C.shape}  | mean={r_C.mean():.4f}  std={r_C.std():.4f}")
print(f"Years saved: {YEAR_START}–{YEAR_END}")