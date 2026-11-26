"""Đọc dữ liệu và tính toán cho 4 mục của dự án (không vẽ ở đây)."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA = Path(__file__).parent / "data"

# ------------------------------------------------------------------ Mục 1
SIGMA, CANH = 0.35, 5          # độ lệch chuẩn của bước đi, hộp [-5, 5]^3


def simulate(seed: int = 1, steps: int = 400, so_hat: int = 3) -> np.ndarray:
    """Bước ngẫu nhiên 3 chiều; chạm thành hộp thì bật ngược lại."""
    rng = np.random.default_rng(seed)
    vi_tri = np.zeros((so_hat, steps, 3))
    vi_tri[:, 0] = rng.uniform(-CANH, CANH, (so_hat, 3))
    for t in range(1, steps):
        moi = vi_tri[:, t - 1] + rng.normal(0, SIGMA, (so_hat, 3))
        moi = np.where(moi > CANH, 2 * CANH - moi, moi)
        moi = np.where(moi < -CANH, -2 * CANH - moi, moi)
        vi_tri[:, t] = moi
    return vi_tri


def msd_curve(steps: int = 400, walks: int = 1000, seed: int = 0):
    """Độ dời bình phương trung bình khi không có thành hộp, và công thức 3*n*sigma^2."""
    rng = np.random.default_rng(seed)
    buoc = rng.normal(0, SIGMA, (walks, steps, 3))
    r2 = (np.cumsum(buoc, axis=1) ** 2).sum(axis=2).mean(axis=0)
    n = np.arange(1, steps + 1)
    return n, r2, 3 * SIGMA ** 2 * n


# ------------------------------------------------------------------ Mục 2
PRICE_TYPES = {"close": "Đóng cửa", "open": "Mở cửa", "high": "Cao nhất",
               "low": "Thấp nhất", "mean": "Trung bình"}
STATUS = ["Trần", "Sàn", "Tăng", "Giảm", "Đứng giá"]


@lru_cache(maxsize=1)
def _load_stock() -> pd.DataFrame:
    df = pd.read_csv(DATA / "abc-16.csv", header=None)        # 365 dòng x 7 mức giá
    bang = pd.DataFrame({"Ngày": pd.date_range("2023-01-01", periods=len(df)),
                         "Mở cửa": df[0], "Cao nhất": df.max(axis=1),
                         "Thấp nhất": df.min(axis=1), "Đóng cửa": df[6],
                         "Trung bình": df.mean(axis=1)})
    bang["Tham chiếu"] = bang["Đóng cửa"].shift(1).fillna(10000)
    c, tc = bang["Đóng cửa"], bang["Tham chiếu"]
    bang["Trạng thái"] = np.select(
        [np.isclose(c, tc * 1.1), np.isclose(c, tc * 0.9), c > tc, c < tc],
        ["Trần", "Sàn", "Tăng", "Giảm"], "Đứng giá")
    so = bang.columns.drop(["Ngày", "Trạng thái"])
    bang[so] = bang[so].round(0)
    return bang


def load_stock() -> pd.DataFrame:
    return _load_stock().copy()


def stock_summary(d: pd.DataFrame, loai: str) -> dict:
    dem = d["Trạng thái"].value_counts()
    g = d[loai]
    return {"so_ngay": int(len(d)), "trung_binh": round(float(g.mean())),
            "do_lech": round(float(g.std())) if len(d) > 1 else 0,
            "cao_nhat": round(float(g.max())), "thap_nhat": round(float(g.min())),
            "dem": {t: int(dem.get(t, 0)) for t in STATUS}}


# ------------------------------------------------------------------ Mục 3
def load_health() -> pd.DataFrame:
    return pd.read_csv(DATA / "health.csv")


@lru_cache(maxsize=1)
def fit_models() -> dict:
    sk = load_health()
    out = {}
    for ten, cot in [("tuoi", ["age"]), ("bmi", ["bmi"]), ("hai", ["age", "bmi"])]:
        m = LinearRegression().fit(sk[cot], sk["chol"])
        out[ten] = {"model": m, "cols": cot,
                    "r2": float(r2_score(sk["chol"], m.predict(sk[cot]))),
                    "coef": [float(x) for x in m.coef_], "intercept": float(m.intercept_)}
    return out


def phan_loai(bmi: float):
    for gioi_han, ten, nguy_co in [(18.5, "Thiếu cân", "Thấp"), (25, "Bình thường", "Trung bình"),
                                   (30, "Thừa cân", "Tăng nhẹ"), (35, "Béo phì độ I", "Vừa"),
                                   (40, "Béo phì độ II", "Nặng")]:
        if bmi < gioi_han:
            return ten, nguy_co
    return "Béo phì độ III", "Rất nặng"


def predict_chol(tuoi: float, cao_cm: float, nang_kg: float) -> dict:
    bmi = nang_kg / (cao_cm / 100) ** 2
    sk = load_health()
    x = pd.DataFrame([[tuoi, bmi]], columns=["age", "bmi"])
    chol = float(fit_models()["hai"]["model"].predict(x)[0])
    ngoai = not (sk["age"].min() <= tuoi <= sk["age"].max()
                 and sk["bmi"].min() <= bmi <= sk["bmi"].max())
    loai, nguy_co = phan_loai(bmi)
    return {"bmi": round(bmi, 1), "loai": loai, "nguy_co": nguy_co,
            "chol": round(chol, 2), "ngoai_pham_vi": ngoai}


# ------------------------------------------------------------------ Mục 4
@lru_cache(maxsize=1)
def _load_gdp() -> pd.DataFrame:
    gdp = pd.read_csv(DATA / "vietnam_gdp_growth_1985_2024.csv")
    lp = pd.read_csv(DATA / "vietnam_inflation_1996_2024.csv")
    kt = gdp.merge(lp, on="Nam")
    kt.columns = ["Năm", "Tăng trưởng GDP (%)", "Lạm phát (%)"]
    return kt


def load_gdp() -> pd.DataFrame:
    return _load_gdp().copy()


def gdp_periods(kt: pd.DataFrame) -> pd.DataFrame:
    g = pd.cut(kt["Năm"], [1995, 2000, 2005, 2010, 2015, 2020, 2024],
               labels=["1996-00", "2001-05", "2006-10", "2011-15", "2016-20", "2021-24"])
    return kt.groupby(g, observed=True)[["Tăng trưởng GDP (%)", "Lạm phát (%)"]].mean()


def gdp_facts(kt: pd.DataFrame) -> dict:
    gd, lp = kt["Tăng trưởng GDP (%)"], kt["Lạm phát (%)"]
    return {"tuong_quan": round(float(gd.corr(lp)), 2),
            "gdp_tb": round(float(gd.mean()), 2), "lp_tb": round(float(lp.mean()), 2),
            "lp_max": float(lp.max()), "lp_max_nam": int(kt.loc[lp.idxmax(), "Năm"]),
            "lp_min": float(lp.min()), "lp_min_nam": int(kt.loc[lp.idxmin(), "Năm"]),
            "gdp_max": float(gd.max()), "gdp_max_nam": int(kt.loc[gd.idxmax(), "Năm"]),
            "gdp_min": float(gd.min()), "gdp_min_nam": int(kt.loc[gd.idxmin(), "Năm"])}
