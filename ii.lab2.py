import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

df_tr_mode = pd.read_csv('D:/univer/proga 4 cem/ii and mmo/train_mode_fill.csv')  # путь к файлу
print(df_tr_mode)  
df_tr_median = pd.read_csv('D:/univer/proga 4 cem/ii and mmo/train_median_fill.csv')  # путь к файлу
print(df_tr_median)  
# df_tr_mean = pd.read_csv('D:/univer/proga 4 cem/ii and mmo/train_mean_fill.csv')  # путь к файлу
# print(df_tr_mean)

#Задача регрессии для моды
X_mode = df_tr_mode.drop(['Transported'], axis=1) #Transported
y_mode = df_tr_mode['Transported']

X_train_mode, X_test_mode, y_train_mode, y_test_mode = train_test_split(X_mode, y_mode, test_size=0.4, random_state=42)
X_test_mode, X_val_mode, y_test_mode, y_val_mode = train_test_split(X_test_mode, y_test_mode, test_size=0.4, random_state=42)

linear_model = LinearRegression() 
linear_model.fit(X_train_mode, y_train_mode)
y_pred_test1 = linear_model.predict(X_test_mode)
MSE = mean_squared_error(y_test_mode, y_pred_test1)
# RMSE = np.sqrt(mean_squared_error(y_test_mode, y_pred_test1))
RMSE = root_mean_squared_error(y_test_mode, y_pred_test1)
MAE = mean_absolute_error(y_test_mode, y_pred_test1)

print(f"Мода: Корень среднеквадратичной ошибки {RMSE}, средняя абсолютная ошибка {MAE}")

l2_linear_model = Ridge(alpha=2.0)
 
l2_linear_model.fit(X_train_mode, y_train_mode)
y_pred_test2 = l2_linear_model.predict(X_test_mode)

MSE_ridge = mean_squared_error(y_test_mode, y_pred_test2)
RMSE_ridge = np.sqrt(MSE_ridge)
MAE_ridge = mean_absolute_error(y_test_mode, y_pred_test2)

print(f"Мода L2 регуляризатор: RMSE={RMSE_ridge:.4f}, MAE={MAE_ridge:.4f}")

# Оценка качества модели на train, test и val для Linear Regression
y_pred_train = linear_model.predict(X_train_mode)
y_pred_val = linear_model.predict(X_val_mode)

rmse_train = np.sqrt(mean_squared_error(y_train_mode, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test_mode, y_pred_test1))
rmse_val = np.sqrt(mean_squared_error(y_val_mode, y_pred_val))

mae_train = mean_absolute_error(y_train_mode, y_pred_train)
mae_test = mean_absolute_error(y_test_mode, y_pred_test1)
mae_val = mean_absolute_error(y_val_mode, y_pred_val)

print("Оценка модели LinearRegression:")
print(f"  Обучающая выборка:     RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}")
print(f"  Тестовая выборка:       RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}")
print(f"  Валидационная выборка:  RMSE = {rmse_val:.4f}, MAE = {mae_val:.4f}")

# Оценка для Ridge regresssion
y_pred_train_ridge = l2_linear_model.predict(X_train_mode)
y_pred_val_ridge = l2_linear_model.predict(X_val_mode)

rmse_train_ridge = np.sqrt(mean_squared_error(y_train_mode, y_pred_train_ridge))
rmse_test_ridge = np.sqrt(mean_squared_error(y_test_mode, y_pred_test2))
rmse_val_ridge = np.sqrt(mean_squared_error(y_val_mode, y_pred_val_ridge))

mae_train_ridge = mean_absolute_error(y_train_mode, y_pred_train_ridge)
mae_test_ridge = mean_absolute_error(y_test_mode, y_pred_test2)
mae_val_ridge = mean_absolute_error(y_val_mode, y_pred_val_ridge)

print("Оценка модели RidgeRegression:")
print(f"Обучающая выборка: RMSE = {rmse_train_ridge:.4f}, MAE = {mae_train_ridge:.4f}")
print(f"Тестовая выборка: RMSE = {rmse_test_ridge:.4f}, MAE = {mae_test_ridge:.4f}")
print(f"Валидационная выборка: RMSE = {rmse_val_ridge:.4f}, MAE = {mae_val_ridge:.4f}")

# Классификация
logreg_model = LogisticRegression(max_iter = 500)
logreg_model.fit(X_train_mode, y_train_mode)
y_pred_test3 = logreg_model.predict(X_test_mode)

accuracy = accuracy_score(y_test_mode, y_pred_test3)
print(f"Оценка работы классификации (доля правильных классификаций моделью) {accuracy}")


report = classification_report(y_test_mode, y_pred_test3)
print(report)

cm = confusion_matrix(y_test_mode, y_pred_test3)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()









# print ("Полиномиальная регрессия для медианы")

# from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split

# X_median = df_tr_median.drop(['Transported'], axis=1)
# y_median = df_tr_median['Transported']

# # Разделим на тренировочные и тестовые данные
# X_train_median, X_test_median, y_train_median, y_test_median = train_test_split(X_median, y_median, test_size=0.4, random_state=42)
# X_test_median, X_val_median, y_test_median, y_val_median = train_test_split(X_test_median, y_test_median, test_size=0.4, random_state=42)

# # Применяем Imputer для заполнения пропущенных значений
# imputer = SimpleImputer(strategy='median')
# X_train_median = imputer.fit_transform(X_train_median)
# X_test_median = imputer.transform(X_test_median)
# X_val_median = imputer.transform(X_val_median)

# # Применяем полиномиальные признаки
# n = 3
# poly_features = PolynomialFeatures(degree=n)
# X_train_median_poly = poly_features.fit_transform(X_train_median)
# X_test_median_poly = poly_features.transform(X_test_median)
# X_val_median_poly = poly_features.transform(X_val_median)

# # Создание модели линейной регрессии
# linear_model = LinearRegression()

# linear_model.fit(X_train_median_poly, y_train_median)

# # Прогноз на тестовой выборке
# y_pred_poly = linear_model.predict(X_test_median_poly)

# MSE_poly = mean_squared_error(y_test_median, y_pred_poly)
# RMSE_poly = np.sqrt(MSE_poly)
# MAE_poly = mean_absolute_error(y_test_median, y_pred_poly)

# print(f"Полиномиальная регрессия (степень {n}): RMSE={RMSE_poly}, MAE={MAE_poly}")

# # Модель с L2 регуляризацией
# l2_linear_model_median = Ridge(alpha=2.0)

# # Обучаем модель с регуляризацией
# l2_linear_model_median.fit(X_train_median_poly, y_train_median)

# # Прогноз на тестовой выборке
# y_pred_test_median = l2_linear_model_median.predict(X_test_median_poly)

# # Вывод результатов с регуляризацией
# print(f"Полиномиальная регрессия (степень {n}) с L2 регуляризатором: RMSE={np.sqrt(mean_squared_error(y_test_median, y_pred_test_median))}, MAE={mean_absolute_error(y_test_median, y_pred_test_median)}")

# plt.show()
