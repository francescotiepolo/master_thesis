"""
Prepares all empirical inputs needed for ProductSpaceModel from three files:
  - hs92_country_product_year_4.csv   (main trade data)
  - product_hs92.csv                  (product lookup)
  - location_country.csv              (country lookup)

Outputs:
  - phi_space.npy       (SP × SP)  proximity matrix
  - beta_C.npy          (SC × SP)  binary RCA matrix
  - alpha_init.npy      (SC × SP)  row-normalised RCA
  - P_init.npy          (SP,)      initial product abundance (world export value)
  - C_init.npy          (SC,)      initial country size (total export value)
  - products_index.csv             product integer index → hs code + name
  - countries_index.csv            country integer index → iso3 + name
  - r_P.npy             (SP,)      intrinsic growth rate per product
  - r_C.npy             (SC,)      intrinsic growth rate per country
"""

import numpy as np
import pandas as pd
from ecomplexity import proximity

# Settings
YEAR = 2000 # Year to use for fitting (2000 used by Hidalgo)
YEAR_END = 2010 # Used to compute growth rates
IN_RANKINGS_ONLY = True # Keep only countries Atlas considers in rankings
TOP_N_PRODUCTS = 200 # Keep only the top N products by world export value

# Load files
print("Loading files...")

trade = pd.read_csv("raw_data/hs92_country_product_year_4.csv", low_memory=False)
products_lookup  = pd.read_csv("raw_data/product_hs92.csv", low_memory=False)
countries_lookup = pd.read_csv("raw_data/location_country.csv", low_memory=False)



# Filter lookup files

# Products: keep only 4-digit level rows (matches trade file)
products_lookup = products_lookup[products_lookup["product_level"] == 4].copy()

# Countries: optionally keep only countries that appear in Atlas rankings
# (removes territories, former countries, tiny islands etc.)
if IN_RANKINGS_ONLY:
    countries_lookup = countries_lookup[countries_lookup["in_rankings"] == True].copy()

print(f"Products in lookup (4-digit): {len(products_lookup)}")
print(f"Countries in lookup:          {len(countries_lookup)}")

valid_countries = countries_lookup["country_iso3_code"].tolist()
valid_products = products_lookup["product_hs92_code"].tolist()



# Filter trade data
def filter_year(trade_df, year):
    ''' Filter trade data for a specific year and keep only valid countries/products with positive export value.
    '''
    df = trade_df[trade_df["year"] == year].copy()
    df = df[df["country_iso3_code"].isin(valid_countries)]
    df = df[df["product_hs92_code"].isin(valid_products)]
    df = df[df["export_value"] > 0].dropna(subset=["export_value"])
    df = df.rename(columns={
        "country_iso3_code": "location_code",
        "product_hs92_code": "hs_product_code"
    })
    return df

df_main = filter_year(trade, YEAR)
df_end = filter_year(trade, YEAR_END)

# Keep only the top N products by world export value
top_products = (df_main.groupby("hs_product_code")["export_value"]
                       .sum()
                       .nlargest(TOP_N_PRODUCTS)
                       .index.tolist())

df_main = df_main[df_main["hs_product_code"].isin(top_products)]
df_end  = df_end[df_end["hs_product_code"].isin(top_products)]
 
trade_cols = {
    "time": "year",
    "loc":  "location_code",
    "prod": "hs_product_code",
    "val":  "export_value"
}



# Build RCA matrix (country × product)
print("Building RCA matrix...")

rca_matrix = df_main.pivot_table(
    index="location_code",
    columns="hs_product_code",
    values="export_rca",
    aggfunc="first"
).fillna(0)

countries = rca_matrix.index.tolist() # List of iso3 codes, length SC
products = rca_matrix.columns.tolist() # List of hs codes, length SP
SC, SP = len(countries), len(products)
print(f"Countries (SC): {SC}, Products (SP): {SP}")



# beta_C — binary RCA matrix 
beta_C = (rca_matrix.values >= 1.0).astype(float) 



# alpha_init — row-normalised RCA
rca_vals = rca_matrix.values.copy()
row_sums = rca_vals.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
alpha_init = rca_vals / row_sums # (SC × SP), rows sum to 1



# P_init — total world export value per product
world_exports = (df_main.groupby("hs_product_code")["export_value"]
                   .sum()
                   .reindex(products)
                   .fillna(0.0)
                   .values.astype(float))

# Normalise
P_init = world_exports / world_exports.mean()
P_init = np.where(P_init == 0, 0.01, P_init) # Avoid zeros



# C_init — total export value per country
total_exports = (df_main.groupby("location_code")["export_value"]
                   .sum()
                   .reindex(countries)
                   .fillna(0.0)
                   .values.astype(float))

# Normalise
C_init = total_exports / total_exports.mean()
C_init = np.where(C_init == 0, 0.01, C_init) # Avoid zeros



# Build index file
# Products index: integer position → hs code + name + category
products_df = pd.DataFrame({
    "position": range(SP),
    "hs_product_code": products
}).merge(
    products_lookup[["product_hs92_code", "product_name_short","product_name"]].rename(columns={
                    "product_hs92_code": "hs_product_code",
    }),
    on="hs_product_code", how="left"
)
products_df.to_csv("extracted_data/products_index.csv", index=False)

# Countries index: integer position → iso3 + name + ranking flag
countries_df = pd.DataFrame({
    "position": range(SC),
    "location_code": countries
}).merge(
    countries_lookup[["country_iso3_code", "country_name", "country_name_short"]].rename(columns={
                    "country_iso3_code": "location_code",
    }),
    on="location_code", how="left"
)
countries_df.to_csv("extracted_data/countries_index.csv", index=False)



# r_P — intrinsic growth rate per product
n_years = YEAR_END - YEAR

world_exports_end = (df_end.groupby("hs_product_code")["export_value"]
                           .sum()
                           .reindex(products)
                           .fillna(0.0)
                           .values.astype(float))

with np.errstate(divide="ignore", invalid="ignore"):
    ratio_P = np.where(world_exports > 0,
                       world_exports_end / world_exports, 1.0)
r_P_raw = ratio_P ** (1.0 / n_years) - 1.0 # Annualised growth rate

# Clip at 1st and 99th percentile of the distribution
p1_P, p99_P = np.percentile(r_P_raw, [1, 99])
clipped_P_mask = (r_P_raw < p1_P) | (r_P_raw > p99_P)
if clipped_P_mask.any():
    print(f"\nr_P clipping bounds: [{p1_P:.4f}, {p99_P:.4f}]")
    print(f"Products clipped ({clipped_P_mask.sum()}):")
    clipped_products_df = products_df[clipped_P_mask].copy()
    clipped_products_df["r_P_raw"] = r_P_raw[clipped_P_mask]
    print(clipped_products_df[["hs_product_code", "product_name_short", "r_P_raw"]].to_string())
r_P = np.clip(r_P_raw, p1_P, p99_P)



# r_C — intrinsic growth rate per country
total_exports_end = (df_end.groupby("location_code")["export_value"]
                           .sum()
                           .reindex(countries)
                           .fillna(0.0)
                           .values.astype(float))

with np.errstate(divide="ignore", invalid="ignore"):
    ratio_C = np.where(total_exports > 0,
                       total_exports_end / total_exports, 1.0)
r_C_raw = ratio_C ** (1.0 / n_years) - 1.0 # Annualised growth rate

p1_C, p99_C = np.percentile(r_C_raw, [1, 99])
clipped_C_mask = (r_C_raw < p1_C) | (r_C_raw > p99_C)
if clipped_C_mask.any():
    print(f"\nr_C clipping bounds: [{p1_C:.4f}, {p99_C:.4f}]")
    print(f"Countries clipped ({clipped_C_mask.sum()}):")
    clipped_countries_df = countries_df[clipped_C_mask].copy()
    clipped_countries_df["r_C_raw"] = r_C_raw[clipped_C_mask]
    print(clipped_countries_df[["location_code", "country_name", "r_C_raw"]].to_string())
r_C = np.clip(r_C_raw, p1_C, p99_C)



# phi_space — product × product proximity matrix
print("Computing proximity matrix...")
prox_df = proximity(df_main, trade_cols) # Returns long format df, with one row per product pair
phi_space = (
    prox_df
    .pivot(index="hs_product_code_1",
           columns="hs_product_code_2",
           values="proximity")
    .reindex(index=products, columns=products)
    .fillna(0.0)
    .values
    .astype(float)
)
np.fill_diagonal(phi_space, 0.0) # Zero diagonal



# Save all arrays
np.save("extracted_data/phi_space.npy", phi_space)
np.save("extracted_data/beta_C.npy", beta_C)
np.save("extracted_data/alpha_init.npy", alpha_init)
np.save("extracted_data/P_init.npy", P_init)
np.save("extracted_data/C_init.npy", C_init)
np.save("extracted_data/r_P.npy", r_P)
np.save("extracted_data/r_C.npy", r_C)



# Print summary
print("\nSummary")
print(f"phi_space   : {phi_space.shape}  | min={phi_space.min():.3f} max={phi_space.max():.3f}")
print(f"beta_C      : {beta_C.shape}  | sparsity={1 - beta_C.mean():.2%}")
print(f"alpha_init  : {alpha_init.shape}")
print(f"P_init      : {P_init.shape}  | min={P_init.min():.3f} max={P_init.max():.3f}")
print(f"C_init      : {C_init.shape}  | min={C_init.min():.3f} max={C_init.max():.3f}")
print(f"r_P         : {r_P.shape}  | mean={r_P.mean():.4f}  std={r_P.std():.4f}")
print(f"r_C         : {r_C.shape}  | mean={r_C.mean():.4f}  std={r_C.std():.4f}")