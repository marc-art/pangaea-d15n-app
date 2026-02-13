import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional mapping backends
_HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

st.set_page_config(page_title="Pangaea Nitrate δ15N Explorer", layout="wide")

# -------------------------
# Config
# -------------------------
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

S_MAP_ALL_BLACK   = 28
S_MAP_STATION_AVG = 70
S_MAP_SELECTED    = 90
S_SCATTER_BIG     = S_MAP_STATION_AVG
S_SECTION_BIG     = S_MAP_STATION_AVG

DEPTH_BINS = [
    ("0–100", (0, 100)),
    ("100–200", (100, 200)),
    ("200–300", (200, 300)),
    ("300–500", (300, 500)),
    ("500–1000", (500, 1000)),
    ("1000–2000", (1000, 2000)),
    ("2000–3000", (2000, 3000)),
    ("3000–4500", (3000, 4500)),
    ("4500–6000", (4500, 6000)),
]

# -------------------------
# Helpers
# -------------------------
def slider_params_from_series(s, default_span=1.0):
    arr = pd.to_numeric(s, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        vmin, vmax = 0.0, float(default_span)
    else:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmin == vmax:
            vmin -= default_span / 2
            vmax += default_span / 2
    return vmin, vmax

def depth_bin_average_station(df_in, zlo, zhi):
    D = df_in[(df_in["Depth"] >= zlo) & (df_in["Depth"] < zhi)].copy()
    if len(D) == 0:
        return D
    return D.groupby("_station_key", as_index=False).agg(
        Lat=("_lat_r", "first"),
        Lon=("_lon_r", "first"),
        d15N=("d15N", "mean"),
        n=("d15N", "size"),
    )

def robust_section_axis(Dc, span_thresh_deg=6.0, min_unique=6, ratio_thresh=1.8):
    lon = Dc["Lon"].to_numpy()
    lat = Dc["Lat"].to_numpy()
    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]
    if lon.size == 0 or lat.size == 0:
        return "Lon", "Longitude"

    lon_span = float(np.nanmax(lon) - np.nanmin(lon))
    lat_span = float(np.nanmax(lat) - np.nanmin(lat))
    lon_u = np.unique(np.round(lon, 3)).size
    lat_u = np.unique(np.round(lat, 3)).size

    if lon_span < span_thresh_deg and lat_span >= span_thresh_deg:
        return "Lat", "Latitude"
    if lat_span < span_thresh_deg and lon_span >= span_thresh_deg:
        return "Lon", "Longitude"

    if (lon_u >= min_unique or lat_u >= min_unique):
        if lon_u >= ratio_thresh * max(lat_u, 1):
            return "Lon", "Longitude"
        if lat_u >= ratio_thresh * max(lon_u, 1):
            return "Lat", "Latitude"

    return ("Lon", "Longitude") if lon_span >= lat_span else ("Lat", "Latitude")

def build_expedition_colors(exp_list):
    cmap = plt.get_cmap("tab20")
    return {ex: cmap(i % cmap.N) for i, ex in enumerate(exp_list)}

def make_map_axes(figsize=(10, 4)):
    if _HAS_CARTOPY:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor="0.92", edgecolor="0.2", linewidth=0.2)
        ax.add_feature(cfeature.OCEAN, facecolor="1.0")
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.35, color="k")
        gl.top_labels = False
        gl.right_labels = False
        return fig, ax, True

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linewidth=0.2, alpha=0.35)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax, False

# -------------------------
# Data loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_pangaea(path: str):
    df = pd.read_csv(path, sep=None, engine="python", dtype={"Expedition": "string"})
    df.columns = df.columns.str.strip()

    if "Latitude" in df.columns and "Lat" not in df.columns:
        df.rename(columns={"Latitude": "Lat"}, inplace=True)
    if "Longitude" in df.columns and "Lon" not in df.columns:
        df.rename(columns={"Longitude": "Lon"}, inplace=True)

    required = ["Expedition", "Lat", "Lon", "Depth", "NO3", "d15N"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()

    for c in ["Lat", "Lon", "Depth", "NO3", "d15N"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Expedition"] = df["Expedition"].astype("string").str.strip()
    df = df[df["Expedition"].notna()]
    df = df[~df["Expedition"].str.lower().isin(["<undefined>", "undefined", "nan", "none", ""])]
    df = df.dropna(subset=["Lat", "Lon", "Depth", "NO3", "d15N"]).reset_index(drop=True)

    # normalize lon to [-180, 180]
    df["Lon"] = ((df["Lon"] + 180) % 360) - 180

    # station key (rounded lat/lon)
    df["_lat_r"] = df["Lat"].round(2)
    df["_lon_r"] = df["Lon"].round(2)
    df["_station_key"] = df["_lat_r"].astype(str) + "_" + df["_lon_r"].astype(str)

    # keep expeditions with >15 stations
    stations_per_exp = df.groupby("Expedition")["_station_key"].nunique()
    valid_exps = stations_per_exp[stations_per_exp > 15].index.tolist()
    df = df[df["Expedition"].isin(valid_exps)].reset_index(drop=True)

    stations_per_exp = df.groupby("Expedition")["_station_key"].nunique().sort_values(ascending=False)

    return df, stations_per_exp

# -------------------------
# UI
# -------------------------
st.title("Pangaea nitrate δ¹⁵N explorer")

with st.sidebar:
    st.header("Data")
    st.caption("Put `Pangaea.csv` in the app folder (or change the path).")
    path = st.text_input("CSV path", value="Pangaea.csv")

try:
    df, stations_per_exp = load_pangaea(path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.write(f"Rows: **{len(df):,}**  |  Expeditions (>15 stations): **{stations_per_exp.size}**")

# -------------------------
# FIGURE 1
# -------------------------
st.subheader("Figure 1")
colA, colB = st.columns([1, 1])

with colA:
    st.markdown("**All data (black)**")
    fig, ax, isC = make_map_axes(figsize=(9, 4))
    if isC:
        ax.scatter(df["Lon"], df["Lat"], s=S_MAP_ALL_BLACK, c="k", alpha=0.6,
                   edgecolor="none", transform=ccrs.PlateCarree())
    else:
        ax.scatter(df["Lon"], df["Lat"], s=S_MAP_ALL_BLACK, c="k", alpha=0.6, edgecolor="none")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

with colB:
    st.markdown("**Station-avg d15N (depth bin)**")
    bin_label = st.selectbox("Depth bin", [b[0] for b in DEPTH_BINS], index=1)
    zlo, zhi = dict(DEPTH_BINS)[bin_label]
    S = depth_bin_average_station(df, zlo, zhi)

    if len(S) == 0:
        st.info("No data in this bin.")
    else:
        vmin, vmax = slider_params_from_series(S["d15N"], default_span=1.0)
        d15_lo, d15_hi = st.slider(
            "d15N range",
            min_value=float(vmin),
            max_value=float(vmax),
            value=(float(vmin), float(vmax)),
        )

        fig, ax, isC = make_map_axes(figsize=(9, 4))
        if isC:
            sc = ax.scatter(S["Lon"], S["Lat"], c=S["d15N"], cmap="viridis",
                            s=S_MAP_STATION_AVG, alpha=0.95, edgecolor="none",
                            vmin=d15_lo, vmax=d15_hi, transform=ccrs.PlateCarree())
        else:
            sc = ax.scatter(S["Lon"], S["Lat"], c=S["d15N"], cmap="viridis",
                            s=S_MAP_STATION_AVG, alpha=0.95, edgecolor="none",
                            vmin=d15_lo, vmax=d15_hi)
        ax.set_title(f"d15N mean {zlo}-{zhi} m (station-avg)")
        plt.colorbar(sc, ax=ax, label="d15N", fraction=0.03, pad=0.02)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

# -------------------------
# FIGURE 2
# -------------------------
st.subheader("Figure 2")

exp_options = stations_per_exp.index.tolist()
default_sel = exp_options[:3]
exps_sel = st.multiselect("Expeditions (only those with >15 stations)", exp_options, default=default_sel)

if len(exps_sel) == 0:
    st.info("Select one or more expeditions.")
    st.stop()

# Filters + axis sliders
lat_vmin, lat_vmax = slider_params_from_series(df["Lat"])
lon_vmin, lon_vmax = slider_params_from_series(df["Lon"])
dep_vmin, dep_vmax = slider_params_from_series(df["Depth"])
no3_vmin, no3_vmax = slider_params_from_series(df["NO3"])
d15_vmin, d15_vmax = slider_params_from_series(df["d15N"])

c1, c2, c3 = st.columns(3)
with c1:
    lat_rng = st.slider("Lat filter", float(lat_vmin), float(lat_vmax), (float(lat_vmin), float(lat_vmax)))
with c2:
    lon_rng = st.slider("Lon filter", float(lon_vmin), float(lon_vmax), (float(lon_vmin), float(lon_vmax)))
with c3:
    dep_rng = st.slider("Depth filter", float(dep_vmin), float(dep_vmax), (float(dep_vmin), float(dep_vmax)))

c4, c5 = st.columns(2)
with c4:
    no3_axis = st.slider("NO3 axis", float(no3_vmin), float(no3_vmax), (float(no3_vmin), float(no3_vmax)))
with c5:
    d15_axis = st.slider("d15N axis", float(d15_vmin), float(d15_vmax), (float(d15_vmin), float(d15_vmax)))

# Subset
D = df[df["Expedition"].isin(exps_sel)].copy()
D = D[
    (D["Lat"].between(*lat_rng)) &
    (D["Lon"].between(*lon_rng)) &
    (D["Depth"].between(*dep_rng))
].copy()

st.write(f"Subset size: **{len(D):,}**")

# Alpha per expedition (for scatters)
st.markdown("**Alpha per expedition (scatter plots)**")
alpha_cols = st.columns(min(3, len(exps_sel)))
alphas = {}
for i, ex in enumerate(exps_sel):
    with alpha_cols[i % len(alpha_cols)]:
        alphas[ex] = st.slider(f"{ex} α", 0.0, 1.0, 0.85, 0.01)

# Colors per expedition
color_map = build_expedition_colors(exps_sel)

# Map + three scatters
map_col, sc_col = st.columns([1.1, 1.4])

with map_col:
    st.markdown("**Map (colored by expedition; labels black)**")
    fig, ax, isC = make_map_axes(figsize=(11, 4.5))
    for ex in exps_sel:
        De = D[D["Expedition"] == ex]
        if len(De) == 0:
            continue
        if isC:
            ax.scatter(De["Lon"], De["Lat"], s=S_MAP_SELECTED, alpha=0.95, edgecolor="none",
                       c=[color_map[ex]], transform=ccrs.PlateCarree())
            ax.text(float(De["Lon"].median()), float(De["Lat"].median()), str(ex),
                    fontsize=8, color="k", ha="center", va="center",
                    transform=ccrs.PlateCarree())
        else:
            ax.scatter(De["Lon"], De["Lat"], s=S_MAP_SELECTED, alpha=0.95, edgecolor="none",
                       c=[color_map[ex]])
            ax.text(float(De["Lon"].median()), float(De["Lat"].median()), str(ex),
                    fontsize=8, color="k", ha="center", va="center")
    ax.set_title("Selected expeditions")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

with sc_col:
    st.markdown("**Scatter panels**")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    for ex in exps_sel:
        De = D[D["Expedition"] == ex]
        if len(De) == 0:
            continue
        a = float(alphas.get(ex, 0.85))

        axs[0].scatter(De["NO3"], De["Depth"], s=S_SCATTER_BIG, alpha=a, edgecolor="none", c=[color_map[ex]])
        axs[1].scatter(De["d15N"], De["Depth"], s=S_SCATTER_BIG, alpha=a, edgecolor="none", c=[color_map[ex]])
        axs[2].scatter(De["NO3"], De["d15N"], s=S_SCATTER_BIG, alpha=a,
                       edgecolor="k", linewidth=0.2, c=[color_map[ex]])

    axs[0].set_title("NO3 vs Depth")
    axs[0].set_xlabel("NO3")
    axs[0].set_ylabel("Depth (m)")
    axs[0].set_xlim(*no3_axis)
    axs[0].set_ylim(dep_rng[1], dep_rng[0])  # surface on top

    axs[1].set_title("d15N vs Depth")
    axs[1].set_xlabel("d15N")
    axs[1].set_ylabel("Depth (m)")
    axs[1].set_xlim(*d15_axis)
    axs[1].set_ylim(dep_rng[1], dep_rng[0])

    axs[2].set_title("NO3 vs d15N")
    axs[2].set_xlabel("NO3")
    axs[2].set_ylabel("d15N")
    axs[2].set_xlim(*no3_axis)
    axs[2].set_ylim(*d15_axis)

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)

# Depth sections + clim sliders
st.markdown("---")
st.subheader("Depth sections (per expedition)")

for ex in exps_sel:
    De = D[D["Expedition"] == ex]
    if len(De) == 0:
        continue

    xvar, xlabel = robust_section_axis(De)

    d15_min, d15_max = slider_params_from_series(De["d15N"])
    no3_min, no3_max = slider_params_from_series(De["NO3"])

    cL, cR = st.columns(2)
    with cL:
        d15_clim = st.slider(f"{ex} d15N range", float(d15_min), float(d15_max),
                             (float(d15_min), float(d15_max)))
    with cR:
        no3_clim = st.slider(f"{ex} NO3 range", float(no3_min), float(no3_max),
                             (float(no3_min), float(no3_max)))

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    fig.suptitle(f"{ex} — Depth sections", fontsize=13)

    sc1 = axs[0].scatter(De[xvar], De["Depth"], c=De["d15N"], cmap="viridis",
                         s=S_SECTION_BIG, edgecolor="none", vmin=d15_clim[0], vmax=d15_clim[1])
    axs[0].set_title(r"$\delta^{15}N$")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("Depth (m)")
    axs[0].set_ylim(float(np.nanmax(De["Depth"])), 0)
    plt.colorbar(sc1, ax=axs[0], label="d15N")

    sc2 = axs[1].scatter(De[xvar], De["Depth"], c=De["NO3"], cmap="viridis",
                         s=S_SECTION_BIG, edgecolor="none", vmin=no3_clim[0], vmax=no3_clim[1])
    axs[1].set_title("NO3")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Depth (m)")
    axs[1].set_ylim(float(np.nanmax(De["Depth"])), 0)
    plt.colorbar(sc2, ax=axs[1], label="NO3")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)

if not _HAS_CARTOPY:
    st.warning(
        "Cartopy not available: maps will plot without coastlines. "
        "On Streamlit Cloud, keep cartopy/pyproj/shapely in requirements."
    )
