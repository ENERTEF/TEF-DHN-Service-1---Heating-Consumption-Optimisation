import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==============================
# CONFIG
# ==============================
DATA_DIR = "data"
OUTPUT_DIR = "output"
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

YEARS = [2020, 2021, 2022, 2023, 2024]

# Nombres EXACTOS que aparecen en la columna "variable"
VAR_MAP = {
    "ENERGIA_F1": "Energía Contador Energía Fase 1 Central",
    "ENERGIA_F2": "Energía Contador Energía Fase 2 Central",
    "POT_GAS": "Potencia Contador Energía Calderas Gas (15 minuto)",
    "POT_F2": "Potencia Contador Energía Fase 2 (15 minuto)",
    "T_RET_F1": "Tª Retorno Contador Energía Fase 1 Central",
    "T_RET_F2": "Tª Retorno Contador Energía Fase 2 Central (15 minuto)",
    "T_IMP_F2": "TEMPERATURA IMPULS FASE II (15 minuto)",
    "T_IMP_GAS1": "Tª Impulsión Caldera gas 1 (15 minuto)",
}

# Variables acumulativas (requieren diff)
CUMULATIVE_KEYS = {"ENERGIA_F1", "ENERGIA_F2", "POT_GAS", "POT_F2"}

# Percentil para limpiar outliers (puedes cambiar alguna a 0.75 si quieres criterio Ángela)
OUTLIER_Q = {k: 0.95 for k in VAR_MAP.keys()}  # por defecto 95%

# ==============================
# HELPERS
# ==============================
def load_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en: {DATA_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower().str.strip()

        # Parse fecha
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        # Parse valor con coma decimal
        df["valor"] = (
            df["valor"]
            .astype(str)
            .str.replace(".", "", regex=False)   # por si hay separador de miles
            .str.replace(",", ".", regex=False)  # coma decimal -> punto
            .str.strip()
        )
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)
    big = big.dropna(subset=["fecha", "valor", "variable"])
    big = big[big["fecha"].dt.year.isin(YEARS)]
    return big


def clean_outliers(s: pd.Series, q: float) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s
    hi = s.quantile(q)
    lo = s.quantile(1 - q)
    return s[(s >= lo) & (s <= hi)]


def compute_mode(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan


def gaussian_plot(s: pd.Series, title: str, outpath: str):
    s = s.dropna()
    if len(s) < 10:
        return

    mu = s.mean()
    sigma = s.std(ddof=1)

    plt.figure()
    plt.hist(s.values, bins=50, density=True)

    x = np.linspace(s.min(), s.max(), 200)
    if sigma and sigma > 0:
        plt.plot(x, norm.pdf(x, mu, sigma))

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def thresholds(s: pd.Series):
    mu = s.mean()
    sd = s.std(ddof=1)
    th2 = mu + 2 * sd
    th3 = mu + 3 * sd
    err2 = ((th2 - mu) / mu) * 100 if mu else np.nan
    err3 = ((th3 - mu) / mu) * 100 if mu else np.nan
    return mu, sd, th2, th3, err2, err3


# ==============================
# MAIN
# ==============================
def main():
    df = load_data()

    # --- Construir series por variable (limpia + diff si acumulativa)
    series_dict = {}
    desc_rows = []

    for key, var_name in VAR_MAP.items():
        sub = df[df["variable"].astype(str).str.strip() == var_name].copy()
        sub = sub.sort_values("fecha")

        if sub.empty:
            print(f"[WARN] No se encontró data para: {var_name}")
            continue

        if key in CUMULATIVE_KEYS:
            sub["value"] = sub["valor"].diff()
            sub.loc[sub["value"] < 0, "value"] = np.nan  # resets
        else:
            sub["value"] = sub["valor"]

        sub = sub.dropna(subset=["fecha", "value"])

        # Limpieza outliers
        q = OUTLIER_Q.get(key, 0.95)
        cleaned = clean_outliers(sub["value"], q=q)
        sub = sub.loc[cleaned.index].copy()

        series_dict[key] = sub[["fecha", "value"]].copy()

        s = sub["value"].dropna()
        if not s.empty:
            desc_rows.append({
                "variable_key": key,
                "variable_name": var_name,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "mode": float(compute_mode(s)),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p05": float(s.quantile(0.05)),
                "p95": float(s.quantile(0.95)),
            })

            gaussian_plot(
                s,
                title=f"Gaussian check - {var_name}",
                outpath=os.path.join(PLOTS_DIR, f"gauss_{key}.png"),
            )

    # --- Guardar tabla descriptiva
    desc_df = pd.DataFrame(desc_rows).sort_values("variable_key")
    desc_df.to_csv(os.path.join(TABLES_DIR, "descriptive_stats_by_variable.csv"), index=False)

    # --- Energía total y Potencia total (suma de fases)
    def merge_on_fecha(a: pd.DataFrame, b: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
        x = a.rename(columns={"value": col_a})
        y = b.rename(columns={"value": col_b})
        return pd.merge(x, y, on="fecha", how="inner")

    energy_m = merge_on_fecha(series_dict["ENERGIA_F1"], series_dict["ENERGIA_F2"], "f1", "f2")
    energy_m["total_energy"] = energy_m["f1"] + energy_m["f2"]

    power_m = merge_on_fecha(series_dict["POT_GAS"], series_dict["POT_F2"], "gas", "f2")
    power_m["total_power"] = power_m["gas"] + power_m["f2"]

    # --- Medias anuales + variación %
    annual = []
    for y in YEARS:
        e_y = energy_m[energy_m["fecha"].dt.year == y]["total_energy"]
        p_y = power_m[power_m["fecha"].dt.year == y]["total_power"]
        annual.append({
            "year": y,
            "avg_total_energy_MWh": float(e_y.mean()) if len(e_y) else np.nan,
            "avg_total_power_MW": float(p_y.mean()) if len(p_y) else np.nan,
        })

    annual_df = pd.DataFrame(annual).sort_values("year")
    annual_df["energy_variation_pct"] = annual_df["avg_total_energy_MWh"].pct_change() * 100
    annual_df["power_variation_pct"] = annual_df["avg_total_power_MW"].pct_change() * 100
    annual_df.to_csv(os.path.join(TABLES_DIR, "annual_means_2020_2024.csv"), index=False)

    # Barras energía/potencia
    plt.figure()
    plt.bar(annual_df["year"].astype(str), annual_df["avg_total_energy_MWh"])
    plt.title("Average Total Energy (MWh) per Year")
    plt.xlabel("Year")
    plt.ylabel("Avg Total Energy (MWh)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bar_avg_energy_2020_2024.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.bar(annual_df["year"].astype(str), annual_df["avg_total_power_MW"])
    plt.title("Average Total Power (MW) per Year")
    plt.xlabel("Year")
    plt.ylabel("Avg Total Power (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bar_avg_power_2020_2024.png"), dpi=160)
    plt.close()

    # --- Umbrales (2std y 3std) sobre series totales
    e_mu, e_sd, e_th2, e_th3, e_err2, e_err3 = thresholds(energy_m["total_energy"])
    p_mu, p_sd, p_th2, p_th3, p_err2, p_err3 = thresholds(power_m["total_power"])

    thr_df = pd.DataFrame([
        {"metric": "mean", "energy_total": e_mu, "power_total": p_mu},
        {"metric": "std", "energy_total": e_sd, "power_total": p_sd},
        {"metric": "threshold_mean_plus_2std", "energy_total": e_th2, "power_total": p_th2},
        {"metric": "threshold_mean_plus_3std", "energy_total": e_th3, "power_total": p_th3},
        {"metric": "threshold_error_pct_2std", "energy_total": e_err2, "power_total": p_err2},
        {"metric": "threshold_error_pct_3std", "energy_total": e_err3, "power_total": p_err3},
    ])
    thr_df.to_csv(os.path.join(TABLES_DIR, "thresholds_energy_power.csv"), index=False)

    # --- Picos mensuales (scatter por año) + conteo
    ENERGY_THRESHOLD = e_th2
    POWER_THRESHOLD = p_th2

    # media mensual
    energy_monthly = energy_m.copy()
    energy_monthly["year"] = energy_monthly["fecha"].dt.year
    energy_monthly["month"] = energy_monthly["fecha"].dt.month
    energy_monthly = energy_monthly.groupby(["year", "month"], as_index=False)["total_energy"].mean()
    energy_monthly["is_peak"] = energy_monthly["total_energy"] > ENERGY_THRESHOLD

    power_monthly = power_m.copy()
    power_monthly["year"] = power_monthly["fecha"].dt.year
    power_monthly["month"] = power_monthly["fecha"].dt.month
    power_monthly = power_monthly.groupby(["year", "month"], as_index=False)["total_power"].mean()
    power_monthly["is_peak"] = power_monthly["total_power"] > POWER_THRESHOLD

    peaks_by_year = []
    for y in YEARS:
        em = energy_monthly[energy_monthly["year"] == y].copy()
        pm = power_monthly[power_monthly["year"] == y].copy()

        peaks_by_year.append({
            "year": y,
            "energy_peaks": int(em["is_peak"].sum()),
            "power_peaks": int(pm["is_peak"].sum())
        })

        # Scatter energía (picos en rojo)
        plt.figure()
        plt.scatter(em["month"], em["total_energy"], label="Monthly mean")
        plt.scatter(em.loc[em["is_peak"], "month"], em.loc[em["is_peak"], "total_energy"], color="red", label="Peak")
        plt.title(f"Monthly Total Energy - {y}")
        plt.xlabel("Month")
        plt.ylabel("Monthly mean total energy")
        plt.xticks(range(1, 13))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"scatter_monthly_energy_{y}.png"), dpi=160)
        plt.close()

        # Scatter potencia (picos en rojo)
        plt.figure()
        plt.scatter(pm["month"], pm["total_power"], label="Monthly mean")
        plt.scatter(pm.loc[pm["is_peak"], "month"], pm.loc[pm["is_peak"], "total_power"], color="red", label="Peak")
        plt.title(f"Monthly Total Power - {y}")
        plt.xlabel("Month")
        plt.ylabel("Monthly mean total power")
        plt.xticks(range(1, 13))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"scatter_monthly_power_{y}.png"), dpi=160)
        plt.close()

    peaks_df = pd.DataFrame(peaks_by_year).sort_values("year")
    peaks_df.to_csv(os.path.join(TABLES_DIR, "peaks_count_by_year.csv"), index=False)

    # barras picos
    plt.figure()
    plt.bar(peaks_df["year"].astype(str), peaks_df["energy_peaks"])
    plt.title("Total Energy Peaks per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of peak months")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bar_energy_peaks_by_year.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.bar(peaks_df["year"].astype(str), peaks_df["power_peaks"])
    plt.title("Total Power Peaks per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of peak months")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bar_power_peaks_by_year.png"), dpi=160)
    plt.close()

    print("DONE ✅")
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
