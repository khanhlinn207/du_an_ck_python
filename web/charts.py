"""Vẽ biểu đồ: seaborn/matplotlib trả PNG, Plotly trả HTML tương tác.

Dùng matplotlib.figure.Figure (không dùng pyplot) để an toàn khi nhiều request chạy cùng lúc.
"""
from functools import lru_cache
from io import BytesIO

import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

import data_service as ds

HONG, XANH, LA, TIM, VANG = "#FF8FAB", "#74C0FC", "#63E6BE", "#B197FC", "#FFD43B"
MAU_TT = {"Trần": TIM, "Sàn": XANH, "Tăng": LA, "Giảm": HONG, "Đứng giá": VANG}
STYLE = {"figure.facecolor": "#FFF5F8", "axes.facecolor": "white", "axes.edgecolor": "#E8A5B8",
         "axes.grid": True, "grid.color": "#F3D6DF", "font.size": 11}


def _fig(w=9, h=4.5):
    fig = Figure(figsize=(w, h))
    fig.patch.set_facecolor("#FFF5F8")
    return fig, fig.subplots()


def _png(fig) -> bytes:
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    return buf.getvalue()


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#F3D6DF")
    for s in ax.spines.values():
        s.set_color("#E8A5B8")


# ------------------------------------------------------------------ Mục 1
def brown_msd_png(steps: int = 400) -> bytes:
    fig, ax = _fig()
    n, r2, ly_thuyet = ds.msd_curve(steps)
    ax.plot(n, r2, color=HONG, lw=3, label="Mô phỏng (1000 đường đi)")
    ax.plot(n, ly_thuyet, "--", color="gray", lw=2, label=r"Lý thuyết $3n\sigma^2$")
    ax.set(xlabel="Số bước n", ylabel=r"Độ dời bình phương trung bình $\langle r^2\rangle$",
           title="Bước ngẫu nhiên 3 chiều: độ dời tăng tuyến tính theo số bước")
    ax.legend()
    _style(ax)
    return _png(fig)


def brown_3d_html(vi_tri) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    for k, (mau, dam) in enumerate([(HONG, "#E64980"), (XANH, "#1C7ED6"), (LA, "#12B886")]):
        p = vi_tri[k]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                                   line=dict(color=mau, width=4), name=f"Hạt {k + 1}"))
        fig.add_trace(go.Scatter3d(x=[p[-1, 0]], y=[p[-1, 1]], z=[p[-1, 2]], mode="markers",
                                   marker=dict(color=dam, size=6), showlegend=False))
    c = ds.CANH
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#FFF5F8",
                      scene=dict(xaxis=dict(range=[-c, c]), yaxis=dict(range=[-c, c]),
                                 zaxis=dict(range=[-c, c]), aspectmode="cube"))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# ------------------------------------------------------------------ Mục 2
def stock_html(d, loai: str, kieu: str) -> str:
    import plotly.graph_objects as go
    fig = go.Figure()
    if kieu == "candle":
        fig.add_trace(go.Candlestick(x=d["Ngày"], open=d["Mở cửa"], high=d["Cao nhất"],
                                     low=d["Thấp nhất"], close=d["Đóng cửa"],
                                     increasing_line_color=LA, decreasing_line_color=HONG,
                                     name="Nến"))
    else:
        fig.add_trace(go.Scatter(x=d["Ngày"], y=d[loai], mode="lines", name=loai,
                                 line=dict(color=HONG, width=3)))
        for tt, mau in MAU_TT.items():
            s = d[d["Trạng thái"] == tt]
            if len(s):
                fig.add_trace(go.Scatter(x=s["Ngày"], y=s[loai], mode="markers", name=tt,
                                         marker=dict(color=mau, size=8,
                                                     line=dict(color="white", width=1))))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#FFF5F8",
                      plot_bgcolor="white", xaxis_rangeslider_visible=False,
                      yaxis_title="Giá (VND)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@lru_cache(maxsize=1)
def stock_month_png() -> bytes:
    bang = ds.load_stock()
    bang["Tháng"] = bang["Ngày"].dt.month
    fig = Figure(figsize=(11, 4.2))
    fig.patch.set_facecolor("#FFF5F8")
    a1, a2 = fig.subplots(1, 2)
    tb = bang.groupby("Tháng")["Đóng cửa"].mean()
    sns.barplot(x=tb.index, y=tb.values, color=HONG, ax=a1)
    a1.set(title="Giá đóng cửa trung bình theo tháng", ylabel="Giá (VND)")
    sns.boxplot(data=bang, x="Tháng", y="Đóng cửa", color=XANH, ax=a2)
    a2.set(title="Phân bố giá đóng cửa theo tháng", ylabel="Giá (VND)")
    for a in (a1, a2):
        _style(a)
    return _png(fig)


# ------------------------------------------------------------------ Mục 3
@lru_cache(maxsize=1)
def health_heatmap_png() -> bytes:
    fig = Figure(figsize=(5.2, 4.2))
    fig.patch.set_facecolor("#FFF5F8")
    ax = fig.subplots()
    pastel = LinearSegmentedColormap.from_list("pastel", ["white", XANH, HONG])
    sns.heatmap(ds.load_health().corr(), annot=True, cmap=pastel, vmin=0, vmax=1,
                linewidths=2, ax=ax)
    ax.set_title("Tương quan giữa tuổi, BMI và cholesterol", fontweight="bold")
    return _png(fig)


@lru_cache(maxsize=1)
def health_regression_png() -> bytes:
    sk, mo = ds.load_health(), ds.fit_models()
    fig = Figure(figsize=(14, 4))
    fig.patch.set_facecolor("#FFF5F8")
    ax = fig.subplots(1, 3)
    for a, cot, key, mau in [(ax[0], "age", "tuoi", HONG), (ax[1], "bmi", "bmi", XANH)]:
        a.scatter(sk[cot], sk["chol"], color=mau, edgecolor="white", s=90)
        xs = np.linspace(sk[cot].min(), sk[cot].max(), 50)
        a.plot(xs, mo[key]["coef"][0] * xs + mo[key]["intercept"], color="gray", lw=2)
        a.set(xlabel=cot, ylabel="chol", title=f"chol theo {cot} (R²={mo[key]['r2']:.3f})")
    du_doan = mo["hai"]["model"].predict(sk[["age", "bmi"]])
    ax[2].scatter(sk["chol"], du_doan, color=TIM, edgecolor="white", s=90)
    ax[2].plot([1.5, 5], [1.5, 5], "--", color="gray")
    ax[2].set(xlabel="chol thực tế", ylabel="chol dự đoán",
              title=f"chol theo age và bmi (R²={mo['hai']['r2']:.3f})")
    for a in ax:
        _style(a)
    return _png(fig)


# ------------------------------------------------------------------ Mục 4
@lru_cache(maxsize=1)
def gdp_line_png() -> bytes:
    kt = ds.load_gdp()
    dai = kt.melt(id_vars="Năm", var_name="Chỉ tiêu", value_name="Giá trị (%)")
    fig, ax = _fig(11, 4.5)
    sns.lineplot(data=dai, x="Năm", y="Giá trị (%)", hue="Chỉ tiêu", marker="o",
                 palette=[LA, HONG], linewidth=2.5, markersize=8, ax=ax)
    ax.axhline(0, color="gray", lw=1)
    ax.set_title("Tăng trưởng GDP và lạm phát của Việt Nam (1996-2024)", fontweight="bold")
    _style(ax)
    return _png(fig)


@lru_cache(maxsize=1)
def gdp_scatter_png() -> bytes:
    fig, ax = _fig(6, 4.5)
    sns.regplot(data=ds.load_gdp(), x="Tăng trưởng GDP (%)", y="Lạm phát (%)",
                scatter_kws={"color": TIM, "s": 80, "edgecolor": "white"},
                line_kws={"color": "gray"}, ax=ax)
    ax.set_title("Lạm phát theo tăng trưởng GDP", fontweight="bold")
    _style(ax)
    return _png(fig)


@lru_cache(maxsize=1)
def gdp_period_png() -> bytes:
    tb = ds.gdp_periods(ds.load_gdp()).reset_index()
    dai = tb.melt(id_vars="Năm", var_name="Chỉ tiêu", value_name="Trung bình (%)")
    fig, ax = _fig(11, 4.5)
    sns.barplot(data=dai, x="Năm", y="Trung bình (%)", hue="Chỉ tiêu", palette=[LA, HONG], ax=ax)
    ax.set(xlabel="Giai đoạn", title="Trung bình theo giai đoạn")
    _style(ax)
    return _png(fig)
