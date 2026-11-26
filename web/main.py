"""Trang web dự án cuối kì lập trình Python nâng cao."""
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import charts
import data_service as ds

# ---------- THÔNG TIN HIỂN THỊ ở trang chủ  ----------
SINH_VIEN = {
    "ten": "Phùng Khánh Linh",
    "mssv": "3110525015",
    "lop": "25CKDL",
    "gvhd": "Thầy Phan Trần Đức Minh",
    "truong": "Trường Đại học Sư phạm - Đại học Đà Nẵng",
    "mon": "Lập trình Python nâng cao",
    "de_tai": "Mô phỏng, phân tích và trực quan hóa dữ liệu bằng Python",
}
MUC = [
    {"url": "/brown", "so": 1, "ten": "Chuyển động Brown",
     "mo_ta": "Mô phỏng bước ngẫu nhiên của các hạt bụi trong hộp kín, truy vết đường đi.",
     "nhan": ["Video", "Biểu đồ 3D"]},
    {"url": "/co-phieu", "so": 2, "ten": "Phân tích giá cổ phiếu",
     "mo_ta": "Bảng điều khiển lọc ngày, loại giá, biểu đồ đường và nến.",
     "nhan": ["Widget", "Biểu đồ"]},
    {"url": "/hoi-quy", "so": 3, "ten": "Hồi quy tuyến tính",
     "mo_ta": "Cholesterol theo tuổi và BMI; widget tính BMI và dự đoán cholesterol.",
     "nhan": ["Widget", "Biểu đồ"]},
    {"url": "/gdp-lam-phat", "so": 4, "ten": "GDP và lạm phát Việt Nam",
     "mo_ta": "Biểu đồ seaborn về lạm phát trong mối liên hệ với tăng trưởng GDP.",
     "nhan": ["Biểu đồ"]},
]
# Địa chỉ dịch vụ Voilà chứa widget (đặt biến môi trường WIDGETS_URL trên Render)
WIDGETS_URL = os.environ.get("WIDGETS_URL", "").rstrip("/")

app = FastAPI(title="Dự án Python nâng cao")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def render(request: Request, name: str, **context):
    context.update(sv=SINH_VIEN, muc=MUC, widgets_url=WIDGETS_URL)
    return templates.TemplateResponse(request=request, name=name, context=context)


def png(data: bytes) -> Response:
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


def so_nguyen(value, mac_dinh, thap, cao):
    """Ép giá trị từ query về số nguyên trong [thap, cao]."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = mac_dinh
    return max(thap, min(cao, value))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render(request, "index.html")


# ------------------------------------------------------------------ Mục 1
@app.get("/brown", response_class=HTMLResponse)
def brown(request: Request, seed: str = "1", steps: str = "400"):
    seed = so_nguyen(seed, 1, 1, 9999)
    steps = so_nguyen(steps, 400, 50, 1000)
    html = charts.brown_3d_html(ds.simulate(seed, steps))
    return render(request, "brown.html", seed=seed, steps=steps, chart3d=html,
                  sigma=ds.SIGMA, canh=ds.CANH)


@app.get("/charts/brown_msd.png")
def chart_brown_msd():
    return png(charts.brown_msd_png())


# ------------------------------------------------------------------ Mục 2
@app.get("/co-phieu", response_class=HTMLResponse)
def co_phieu(request: Request, tu: str = "1", den: str = "100",
             loai: str = "close", kieu: str = "line"):
    tu, den = so_nguyen(tu, 1, 1, 365), so_nguyen(den, 100, 1, 365)
    if tu > den:
        tu, den = den, tu
    loai = loai if loai in ds.PRICE_TYPES else "close"
    kieu = kieu if kieu in ("line", "candle") else "line"
    ten_loai = ds.PRICE_TYPES[loai]
    d = ds.load_stock().iloc[tu - 1:den]
    bang = d.head(15).copy()
    bang["Ngày"] = bang["Ngày"].dt.strftime("%d/%m/%Y")
    return render(request, "co_phieu.html", tu=tu, den=den, loai=loai, kieu=kieu,
                  loai_ds=ds.PRICE_TYPES, ten_loai=ten_loai, tom_tat=ds.stock_summary(d, ten_loai),
                  chart=charts.stock_html(d, ten_loai, kieu),
                  bang=bang.to_html(index=False, border=0, classes="bang", justify="left"))


@app.get("/charts/co_phieu_thang.png")
def chart_stock_month():
    return png(charts.stock_month_png())


# ------------------------------------------------------------------ Mục 3
@app.get("/hoi-quy", response_class=HTMLResponse)
def hoi_quy(request: Request, tuoi: str = "30", cao: str = "165", nang: str = "55"):
    tuoi, cao, nang = so_nguyen(tuoi, 30, 1, 120), so_nguyen(cao, 165, 100, 220), so_nguyen(nang, 55, 20, 250)
    mo = ds.fit_models()
    cac_mo_hinh = [("Cholesterol theo tuổi", mo["tuoi"]), ("Cholesterol theo BMI", mo["bmi"]),
                   ("Cholesterol theo tuổi và BMI", mo["hai"])]
    return render(request, "hoi_quy.html", tuoi=tuoi, cao=cao, nang=nang,
                  kq=ds.predict_chol(tuoi, cao, nang), cac_mo_hinh=cac_mo_hinh)


@app.get("/charts/tuong_quan.png")
def chart_corr():
    return png(charts.health_heatmap_png())


@app.get("/charts/hoi_quy.png")
def chart_regression():
    return png(charts.health_regression_png())


# ------------------------------------------------------------------ Mục 4
@app.get("/gdp-lam-phat", response_class=HTMLResponse)
def gdp_lam_phat(request: Request):
    kt = ds.load_gdp()
    return render(request, "gdp_lam_phat.html", f=ds.gdp_facts(kt),
                  giai_doan=ds.gdp_periods(kt).round(2).reset_index().rename(columns={"Năm": "Giai đoạn"})
                  .to_html(index=False, border=0, classes="bang", justify="left"))


@app.get("/charts/gdp_duong.png")
def chart_gdp_line():
    return png(charts.gdp_line_png())


@app.get("/charts/gdp_tuong_quan.png")
def chart_gdp_scatter():
    return png(charts.gdp_scatter_png())


@app.get("/charts/gdp_giai_doan.png")
def chart_gdp_period():
    return png(charts.gdp_period_png())
