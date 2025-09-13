import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# Datasetni yuklash
df = pd.read_csv("train.csv")

# Foydalanadigan ustunlar
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF"]
target = "SalePrice"

X = df[features]
y = df[target]

# Bo'sh qiymatlarni 0 bilan to'ldiramiz
X = X.fillna(0)

# Train-test bo'linishi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model yaratish va o'qitish
model = LinearRegression()
model.fit(X_train, y_train)
# modelni saqlash
joblib.dump(model, "house_price_model.pkl")

# Bashorat qilish
y_pred = model.predict(X_test)

# Natijani baholash
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
print("Birinchi 5 ta bashorat:", y_pred[:5])









# 1. Dataset hajmi va ustunlar
print("Dataset shakli:", df.shape)
print("Ustunlar:", df.columns.tolist())

# 2. Bo‘sh qiymatlar soni
print("\nBo‘sh qiymatlar:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# 3. Statistik ma’lumot
print("\nRaqamli ustunlarning statistikasi:")
print(df.describe())

# 4. Narx taqsimoti (grafik)
plt.figure(figsize=(8,5))
sns.histplot(df["SalePrice"], kde=True)
plt.title("Uy narxlari taqsimoti")
plt.show()

# 5. Bog‘liqlik (GrLivArea va SalePrice)
plt.figure(figsize=(8,5))
sns.scatterplot(x="GrLivArea", y="SalePrice", data=df)
plt.title("Yashash maydoni va uy narxi bog‘liqligi")
plt.show()
