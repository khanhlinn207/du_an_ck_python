"""Widget 2: tính BMI và dự đoán cholesterol (ipywidgets + scikit-learn + Voilà)."""
from pathlib import Path

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA = Path(__file__).parent / "data" / "health.csv"


def phan_loai(bmi):
    for gioi_han, ten, nguy_co in [(18.5, "Thiếu cân", "Thấp"), (25, "Bình thường", "Trung bình"),
                                   (30, "Thừa cân", "Tăng nhẹ"), (35, "Béo phì độ I", "Vừa"),
                                   (40, "Béo phì độ II", "Nặng")]:
        if bmi < gioi_han:
            return ten, nguy_co
    return "Béo phì độ III", "Rất nặng"


def train():
    """Huấn luyện mô hình chol theo (age, bmi) trên health.csv (18 dòng, rất nhanh)."""
    sk = pd.read_csv(DATA)
    mo_hinh = LinearRegression().fit(sk[["age", "bmi"]], sk["chol"])
    r2 = r2_score(sk["chol"], mo_hinh.predict(sk[["age", "bmi"]]))
    return sk, mo_hinh, r2


def create_app():
    sk, m_hai, r2 = train()

    # ----- Widget 1: tính BMI -----
    cao1 = widgets.FloatText(value=165, description="Cao (cm):")
    nang1 = widgets.FloatText(value=55, description="Nặng (kg):")
    nut_bmi = widgets.Button(description="Tính BMI", button_style="success")
    out_bmi = widgets.Output()

    def tinh_bmi(_=None):
        with out_bmi:
            clear_output(wait=True)
            if cao1.value <= 0 or nang1.value <= 0:
                print("Chiều cao và cân nặng phải lớn hơn 0.")
                return
            bmi = nang1.value / (cao1.value / 100) ** 2
            ten, nguy_co = phan_loai(bmi)
            print(f"BMI = {bmi:.1f} kg/m2 -> {ten} (nguy cơ sức khỏe: {nguy_co})")

    nut_bmi.on_click(tinh_bmi)

    # ----- Widget 2: dự đoán cholesterol -----
    cao2 = widgets.FloatText(value=165, description="Cao (cm):")
    nang2 = widgets.FloatText(value=55, description="Nặng (kg):")
    tuoi2 = widgets.IntText(value=30, description="Tuổi:")
    nut_chol = widgets.Button(description="Dự đoán cholesterol", button_style="success")
    out_chol = widgets.Output()

    def du_doan_chol(_=None):
        with out_chol:
            clear_output(wait=True)
            if cao2.value <= 0 or nang2.value <= 0 or tuoi2.value <= 0:
                print("Chiều cao, cân nặng và tuổi phải lớn hơn 0.")
                return
            bmi = nang2.value / (cao2.value / 100) ** 2            # bước 1: tính BMI
            x = pd.DataFrame([[tuoi2.value, bmi]], columns=["age", "bmi"])
            chol = m_hai.predict(x)[0]                             # bước 2: mô hình chol(age, bmi)
            print(f"BMI = {bmi:.1f} ({phan_loai(bmi)[0]}) -> cholesterol dự đoán = {chol:.2f}")
            if not (sk["age"].min() <= tuoi2.value <= sk["age"].max() and
                    sk["bmi"].min() <= bmi <= sk["bmi"].max()):
                print("Lưu ý: giá trị nằm ngoài phạm vi dữ liệu huấn luyện nên kết quả kém tin cậy.")

    nut_chol.on_click(du_doan_chol)
    tinh_bmi()
    du_doan_chol()
    return widgets.VBox([
        widgets.HTML("<h2>Hồi quy tuyến tính: tuổi, BMI và cholesterol</h2>"
                     f"<p>Mô hình chol = a·age + b·bmi + c huấn luyện trên health.csv "
                     f"(18 quan sát), R² = {r2:.3f}.</p>"),
        widgets.HTML("<h3>Tính chỉ số BMI</h3>"), cao1, nang1, nut_bmi, out_bmi,
        widgets.HTML("<h3>Dự đoán cholesterol</h3>"), cao2, nang2, tuoi2, nut_chol, out_chol])
