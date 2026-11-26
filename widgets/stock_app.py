"""Widget 1: bảng điều khiển phân tích giá cổ phiếu (ipywidgets + Voilà)."""
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display

DATA = Path(__file__).parent / "data" / "abc-16.csv"
HONG, XANH, LA, TIM, VANG = "#FF8FAB", "#74C0FC", "#63E6BE", "#B197FC", "#FFD43B"
MAU_TT = {"Trần": TIM, "Sàn": XANH, "Tăng": LA, "Giảm": HONG, "Đứng giá": VANG}
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.facecolor": "#FFF5F8", "axes.edgecolor": "#E8A5B8",
                     "grid.color": "#F3D6DF", "font.size": 11})


def load_table() -> pd.DataFrame:
    """Đọc abc-16.csv và lập bảng giá theo ngày (giống bài làm trên Colab)."""
    df = pd.read_csv(DATA, header=None)                       # 365 dòng x 7 mức giá
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


def ve_hinh(ax, d, loai, kieu):
    if kieu == "Đường":
        ax.plot(d["Ngày"], d[loai], color=HONG, lw=2.5)
        for tt in MAU_TT:
            s = d[d["Trạng thái"] == tt]
            ax.scatter(s["Ngày"], s[loai], color=MAU_TT[tt], edgecolor="white", s=45,
                       label=tt, zorder=3)
        ax.legend(ncol=5, fontsize=9)
    elif kieu == "Nến":
        mau = d["Trạng thái"].map(MAU_TT)
        ax.vlines(d["Ngày"], d["Thấp nhất"], d["Cao nhất"], color=mau, lw=1)
        ax.bar(d["Ngày"], (d["Đóng cửa"] - d["Mở cửa"]).abs().clip(lower=30),
               bottom=d[["Mở cửa", "Đóng cửa"]].min(axis=1), color=mau, width=0.7)
    else:
        sns.boxplot(data=d.assign(Tháng=d["Ngày"].dt.month), x="Tháng", y=loai,
                    color=XANH, ax=ax)
    ax.set_ylabel("Giá (VND)")
    ax.set_title(kieu if kieu == "Nến" else f"{kieu} - {loai}", fontweight="bold")


def create_app():
    bang = load_table()

    # ----- Module 1: bộ lọc và nút điều khiển -----
    w_ngay = widgets.IntRangeSlider(value=[1, 100], min=1, max=365, description="Ngày thứ:",
                                    continuous_update=False,
                                    layout=widgets.Layout(width="500px"))
    w_gia = widgets.Dropdown(options=["Đóng cửa", "Mở cửa", "Cao nhất", "Thấp nhất", "Trung bình"],
                             value="Đóng cửa", description="Loại giá:")
    w_kieu = widgets.ToggleButtons(options=["Đường", "Nến", "Hộp theo tháng"], description="Biểu đồ:")
    w_tt = widgets.SelectMultiple(options=list(MAU_TT), value=list(MAU_TT),
                                  description="Trạng thái:", rows=5)
    nut_xem = widgets.Button(description="Cập nhật", button_style="success")
    nut_reset = widgets.Button(description="Đặt lại", button_style="warning")
    out_bang, out_hinh = widgets.Output(), widgets.Output()   # module 2 và module 3

    # ----- Module 2 (bảng) và Module 3 (biểu đồ) -----
    def cap_nhat(_=None):
        a, b = w_ngay.value
        d = bang.iloc[a - 1:b]
        bt = d[d["Trạng thái"].isin(w_tt.value)]
        with out_bang:
            clear_output(wait=True)
            display(bt.head(10))
            g = d[w_gia.value]
            print(f"{len(bt)} ngày phù hợp | {w_gia.value}: TB {g.mean():.0f}, "
                  f"độ lệch chuẩn {g.std():.0f}, cao nhất {g.max():.0f}, thấp nhất {g.min():.0f}")
        with out_hinh:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ve_hinh(ax, d, w_gia.value, w_kieu.value)
            plt.show()
            plt.close(fig)

    def dat_lai(_=None):
        w_ngay.value, w_gia.value, w_kieu.value = [1, 100], "Đóng cửa", "Đường"
        w_tt.value = list(MAU_TT)
        cap_nhat()

    nut_xem.on_click(cap_nhat)
    nut_reset.on_click(dat_lai)

    giao_dien = widgets.VBox([
        widgets.HTML("<h2>Bảng điều khiển phân tích giá cổ phiếu abc-16</h2>"),
        widgets.HTML("<h3>1. Bộ lọc và nút điều khiển</h3>"),
        widgets.HBox([widgets.VBox([w_ngay, w_gia, w_kieu]), w_tt]),
        widgets.HBox([nut_xem, nut_reset]),
        widgets.HTML("<h3>2. Bảng số liệu</h3>"), out_bang,
        widgets.HTML("<h3>3. Biểu đồ</h3>"), out_hinh])
    cap_nhat()                                                # hiện sẵn kết quả lần đầu
    return giao_dien
