import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Saqlangan modelni yuklash
model = joblib.load("house_price_model.pkl")

st.title("Uy narxini bashorat qilish")

# Foydalanuvchi kiritadigan inputlar (misol uchun eng asosiy ustunlar)
gr_liv_area = st.number_input("Yashash maydoni (GrLivArea)", min_value=100, max_value=5000, value=1500)
overall_qual = st.number_input("Umumiy sifat (OverallQual)", min_value=1, max_value=10, value=5)
year_built = st.number_input("Uy qurilgan yil (YearBuilt)", min_value=1800, max_value=2025, value=2000)

# Model uchun dataframe yasaymiz
input_df = pd.DataFrame({
    "GrLivArea": [gr_liv_area],
    "OverallQual": [overall_qual],
    "YearBuilt": [year_built]
})

if st.button("Bashorat qilish"):
    prediction = model.predict(input_df)
    st.success(f"Uy narxi taxminan: ${prediction[0]:,.0f}")
    
# Datasetni o‘qish
df = pd.read_csv("train.csv")

# Faqat asosiy ustunlar
features = ["GrLivArea", "OverallQual", "YearBuilt", "GarageCars", "TotalBsmtSF"]
X = df[features]
y = df["SalePrice"]

# Train/Test bo‘lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Bashorat
y_pred = model.predict(X_test)

# Streamlit interface
st.title("Uy narxi bashorati 🏠")
st.write("Oddiy ML modeli yordamida uy narxini hisoblab beramiz.")

# Model aniqligi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.metric("Model aniqligi (R²)", f"{r2:.2f}")
st.metric("O‘rtacha xatolik (MSE)", f"{mse:,.0f}")
# Grafikni chizamiz
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(y_test, y_pred, alpha=0.5, label="Bashorat qilingan qiymatlar")
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red", lw=2, label="Mukammal chiziq (y=x)")
ax.set_xlabel("Haqiqiy narx")
ax.set_ylabel("Bashorat qilingan narx")
ax.set_title("Haqiqiy va Bashorat qilingan qiymatlar solishtirishi")
ax.legend()
ax.grid(True)

# Streamlitda chiqaramiz
st.pyplot(fig)


# Foydalanuvchi kiritishi uchun inputlar
st.subheader("🔮 Sizning uyingiz narxini taxmin qilish")
grlivarea = st.number_input("Yashash maydoni (GrLivArea)", min_value=200, max_value=6000, value=1500)
overallqual = st.slider("Sifat bahosi (OverallQual)", 1, 10, 5)
yearbuilt = st.number_input("Qurilgan yil (YearBuilt)", min_value=1800, max_value=2025, value=2000)
garagecars = st.slider("Garaj mashina sig‘imi (GarageCars)", 0, 5, 2)
totalbsmt = st.number_input("Podval maydoni (TotalBsmtSF)", min_value=0, max_value=3000, value=800)

if st.button("Narxni hisoblash"):
    prediction = model.predict([[grlivarea, overallqual, yearbuilt, garagecars, totalbsmt]])[0]
    st.success(f"Taxminiy uy narxi: ${int(prediction):,}")





