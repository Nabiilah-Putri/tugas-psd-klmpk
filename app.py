import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Dashboard Film & Kriminalitas ASEAN", layout="wide")

ASEAN_COUNTRIES = [
    "Indonesia",
    "Malaysia",
    "Philippines",
    "Brunei Darussalam",
    "Singapore",
    "Thailand",
    "Myanmar",
    "Cambodia",
    "Vietnam",
]

COUNTRY_FIX_MAP = {
    # umumkan typo dari data kriminalitas
    "Philipina": "Philippines",
    "Philippine": "Philippines",
    "Myannar": "Myanmar",
    "Brunei": "Brunei Darussalam",
    # hilangkan spasi aneh
    " Indonesia": "Indonesia",
    "Malaysia ": "Malaysia",
    " Philipina": "Philippines",
}

GENRES_LIST = [
    "Drama", "Horror", "Thriller", "Mystery", "Action", "Documentary",
    "Science Fiction", "Comedy", "Crime", "Adventure", "War", "Romance",
    "History", "Family", "Music", "Fantasy", "Animation", "TV Movie", "Western"
]

# -------------------------------
# HELPERS
# -------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\xa0", " ").replace("  ", " ") for c in df.columns]
    return df

def standardize_country_name(x: str) -> str:
    if pd.isna(x):
        return x
    name = str(x).strip()
    name = COUNTRY_FIX_MAP.get(name, name)
    return name

@st.cache_data
def load_film(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = normalize_columns(df)

    # Pastikan kolom tanggal ada dan bertipe datetime
    if "release_date" not in df.columns and "release date" in df.columns:
        df = df.rename(columns={"release date": "release_date"})
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["Tahun"] = df["release_date"].dt.year
    else:
        # Jika tidak ada, buat Tahun dari NaN
        df["Tahun"] = np.nan

    # Pastikan kolom genres ada
    if "genres" not in df.columns and "genres " in df.columns:
        df = df.rename(columns={"genres ": "genres"})
    # Pastikan kolom production_countries ada
    if "production_countries" not in df.columns and "production_countries " in df.columns:
        df = df.rename(columns={"production_countries ": "production_countries"})

    # Ambil negara pertama dari daftar production_countries
    def first_country(s):
        if pd.isna(s):
            return np.nan
        # Pecah, strip spasi
        parts = [p.strip() for p in str(s).split(",")]
        return parts[0] if parts else np.nan

    df["Negara"] = df["production_countries"].apply(first_country)
    df["Negara"] = df["Negara"].apply(standardize_country_name)

    # Filter hanya ASEAN
    df = df[df["Negara"].isin(ASEAN_COUNTRIES)]

    return df

@st.cache_data
def load_crime(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)

    # Hapus kolom Safety Rate kalau ada
    if "Safety Rate" in df.columns:
        df = df.drop(columns=["Safety Rate"])

    # Normalisasi nama kolom inti
    for col_candidate in ["Negara", " Negara "]:
        if col_candidate in df.columns and col_candidate != "Negara":
            df = df.rename(columns={col_candidate: "Negara"})
    for col_candidate in ["Tahun", " Tahun "]:
        if col_candidate in df.columns and col_candidate != "Tahun":
            df = df.rename(columns={col_candidate: "Tahun"})
    for col_candidate in ["Crime Rate", " Crime Rate"]:
        if col_candidate in df.columns and col_candidate != "Crime Rate":
            df = df.rename(columns={col_candidate: "Crime Rate"})

    # Bersihkan nilai negara
    df["Negara"] = df["Negara"].astype(str).apply(standardize_country_name)

    # Konversi Tahun ke int
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")

    # Konversi Crime Rate dari koma ke titik (e.g., "37,9" -> 37.9)
    def to_float_with_comma(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace(" ", "").replace(",", ".")
        try:
            return float(s)
        except:
            return np.nan

    df["Crime Rate"] = df["Crime Rate"].apply(to_float_with_comma)

    # Filter ASEAN
    df = df[df["Negara"].isin(ASEAN_COUNTRIES)]

    return df

def explode_genres(df_film: pd.DataFrame) -> pd.DataFrame:
    # Pecah genre menjadi baris
    g = df_film[["Negara", "Tahun", "genres"]].dropna()
    g = g.assign(genres=g["genres"].astype(str))
    g = g.assign(genre_split=g["genres"].str.split(","))
    g = g.explode("genre_split")
    g["genre_split"] = g["genre_split"].str.strip()
    return g.rename(columns={"genre_split": "Genre"})

def one_hot_genres(df_film: pd.DataFrame) -> pd.DataFrame:
    # One-hot untuk 19 genre yang didefinisikan
    df = df_film.copy()
    df["genres"] = df["genres"].fillna("")
    for g in GENRES_LIST:
        df[g] = df["genres"].apply(lambda x: 1 if g in str(x) else 0)
    # Agregasi per negara-tahun
    agg = df.groupby(["Negara", "Tahun"], dropna=True)[GENRES_LIST].sum().reset_index()
    return agg

def merge_film_crime_counts(df_film: pd.DataFrame, df_crime: pd.DataFrame) -> pd.DataFrame:
    # Hitung jumlah film per negara-tahun
    film_counts = (
        df_film.groupby(["Negara", "Tahun"], dropna=True)
        .size()
        .reset_index(name="Jumlah Film")
    )
    merged = film_counts.merge(df_crime, on=["Negara", "Tahun"], how="inner")
    return merged

def ols_by_genre(df_film: pd.DataFrame, df_crime: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    # Siapkan fitur genre one-hot dan gabungkan dengan crime
    genre_agg = one_hot_genres(df_film)
    df = genre_agg.merge(df_crime[["Negara", "Tahun", "Crime Rate"]], on=["Negara", "Tahun"], how="inner").dropna()
    if df.empty:
        return None
    X = df[GENRES_LIST]
    y = df["Crime Rate"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

# -------------------------------
# UI
# -------------------------------
st.title("🎬 Dashboard Film & Kriminalitas ASEAN (2020–2024)")
st.caption("Interaktif: jelajahi jumlah film, distribusi genre, dan tren Crime Rate per negara & tahun.")

# Sidebar: file paths
st.sidebar.header("Data")
film_path = "film_asean_2020_2024.xls"
crime_path = "data_kriminalitas.csv"

# Load data
try:
    film_df = load_film(film_path)
except Exception as e:
    st.error(f"Gagal memuat data film: {e}")
    st.stop()

try:
    crime_df = load_crime(crime_path)
except Exception as e:
    st.error(f"Gagal memuat data kriminalitas: {e}")
    st.stop()

# Tahun yang tersedia dari kedua sumber
years_film = sorted([y for y in film_df["Tahun"].dropna().unique() if pd.notna(y)])
years_crime = sorted([int(y) for y in crime_df["Tahun"].dropna().unique() if pd.notna(y)])
years_common = sorted(list(set(years_film) & set(years_crime)))
if not years_common:
    st.warning("Tidak ada tahun yang overlap antara data film dan kriminalitas.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter")
selected_country = st.sidebar.selectbox("Pilih Negara", sorted(ASEAN_COUNTRIES))
year_min, year_max = min(years_common), max(years_common)
selected_years = st.sidebar.slider("Rentang Tahun", min_value=year_min, max_value=year_max, value=(year_min, year_max))

# Filter data sesuai pilihan
film_sel = film_df[(film_df["Negara"] == selected_country) & (film_df["Tahun"].between(selected_years[0], selected_years[1]))]
crime_sel = crime_df[(crime_df["Negara"] == selected_country) & (crime_df["Tahun"].between(selected_years[0], selected_years[1]))]

# KPI
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Negara", selected_country)
with col2:
    st.metric("Periode", f"{selected_years[0]}–{selected_years[1]}")
with col3:
    total_films = len(film_sel)
    st.metric("Total Film (periode terpilih)", f"{total_films}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Distribusi Genre", "Tren Per Tahun", "Crime Rate vs Jumlah Film", "Regresi OLS"])

# -------------------------------
# TAB 1: Distribusi Genre
# -------------------------------
with tab1:
    st.subheader(f"Distribusi Genre — {selected_country} ({selected_years[0]}–{selected_years[1]})")
    if film_sel.empty:
        st.info("Tidak ada data film untuk filter saat ini.")
    else:
        genres_exploded = explode_genres(film_sel)
        genre_counts = (
            genres_exploded.groupby("Genre")
            .size()
            .reset_index(name="Jumlah")
            .sort_values("Jumlah", ascending=False)
        )
        fig = px.bar(genre_counts, x="Genre", y="Jumlah", color="Jumlah",
                     title="Distribusi Genre Film",
                     color_continuous_scale="Viridis")
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)

        # Tabel contoh film (opsional)
        with st.expander("Lihat sampel film"):
            cols_show = ["title", "Tahun", "genres", "Negara"]
            cols_show = [c for c in cols_show if c in film_sel.columns]
            st.dataframe(film_sel[cols_show].sort_values("Tahun", ascending=False).head(50), use_container_width=True)

# -------------------------------
# TAB 2: Tren Per Tahun
# -------------------------------
with tab2:
    st.subheader(f"Tren Jumlah Film & Crime Rate — {selected_country}")
    # Jumlah film per tahun
    if film_sel.empty and crime_sel.empty:
        st.info("Tidak ada data untuk ditampilkan pada filter saat ini.")
    else:
        film_year_counts = (
            film_sel.groupby("Tahun")
            .size()
            .reset_index(name="Jumlah Film")
        )
        crime_year = crime_sel[["Tahun", "Crime Rate"]].dropna().sort_values("Tahun")

        # Gabungkan untuk mempermudah plot dua sumbu
        trend_df = pd.merge(film_year_counts, crime_year, on="Tahun", how="outer").sort_values("Tahun")
        trend_df["Jumlah Film"] = trend_df["Jumlah Film"].fillna(0)

        fig = px.bar(trend_df, x="Tahun", y="Jumlah Film", title="Jumlah Film per Tahun")
        fig.update_layout(yaxis_title="Jumlah Film")
        st.plotly_chart(fig, use_container_width=True)

        if not crime_year.empty:
            fig2 = px.line(trend_df, x="Tahun", y="Crime Rate", markers=True, title="Crime Rate per Tahun")
            fig2.update_layout(yaxis_title="Crime Rate")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Data Crime Rate tidak tersedia untuk rentang tahun ini.")

# -------------------------------
# TAB 3: Crime Rate vs Jumlah Film
# -------------------------------
with tab3:
    st.subheader(f"Korelasi Sederhana — {selected_country}")
    merged_counts = merge_film_crime_counts(film_df[film_df["Negara"] == selected_country], crime_df[crime_df["Negara"] == selected_country])
    merged_counts = merged_counts[merged_counts["Tahun"].between(selected_years[0], selected_years[1])].dropna(subset=["Crime Rate"])

    if merged_counts.empty:
        st.info("Tidak ada pasangan data (Jumlah Film, Crime Rate) untuk filter saat ini.")
    else:
        fig = px.scatter(
            merged_counts,
            x="Jumlah Film",
            y="Crime Rate",
            color="Tahun",
            title="Scatter: Jumlah Film vs Crime Rate (diwarnai per Tahun)",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabel ringkas
        st.dataframe(merged_counts.sort_values("Tahun"), use_container_width=True)

# -------------------------------
# TAB 4: Regresi OLS (Genre → Crime Rate)
# -------------------------------
with tab4:
    st.subheader("Regresi OLS: Apakah jumlah produksi per genre memprediksi Crime Rate?")
    results = ols_by_genre(film_df, crime_df)
    if results is None:
        st.info("Data tidak cukup untuk menjalankan regresi.")
    else:
        # Ringkasan metrik utama
        r2 = results.rsquared
        r2_adj = results.rsquared_adj
        f_p = results.f_pvalue
        colA, colB, colC = st.columns(3)
        colA.metric("R-squared", f"{r2:.3f}")
        colB.metric("Adjusted R-squared", f"{r2_adj:.3f}")
        colC.metric("Prob(F-statistic)", f"{f_p:.3f}")

        # Koefisien
        params = results.params.drop("const")
        conf_int = results.conf_int().drop(index="const")
        coef_df = pd.DataFrame({
            "Genre": params.index,
            "Coefficient": params.values,
            "Lower CI": conf_int[0].values,
            "Upper CI": conf_int[1].values,
            "p-value": results.pvalues.drop("const").values
        }).sort_values("Coefficient", ascending=False)

        fig_coef = px.bar(
            coef_df,
            x="Genre",
            y="Coefficient",
            error_y=coef_df["Upper CI"] - coef_df["Coefficient"],
            error_y_minus=coef_df["Coefficient"] - coef_df["Lower CI"],
            color="p-value",
            color_continuous_scale="RdBu_r",
            title="Koefisien Regresi per Genre (95% CI)",
        )
        fig_coef.update_layout(xaxis={'categoryorder': 'category ascending'})
        st.plotly_chart(fig_coef, use_container_width=True)

        with st.expander("Lihat tabel koefisien & p-value"):
            st.dataframe(coef_df, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Catatan: Dashboard ini menampilkan hubungan deskriptif. Hasil regresi sebelumnya pada tugas menunjukkan produksi film tidak mempengaruhi Crime Rate secara signifikan (p-values besar, R² rendah).")