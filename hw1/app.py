import joblib
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phik
import seaborn as sns
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="Car price predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Car price prediction")
st.write("Необходимо указать параметры автомобиля или загрузить csv")


# Загрузка модели и препроцессора
@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open(os.path.join(MODELS_DIR, 'median_brand.pkl'), 'rb') as f:
        median_brand = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'median_global.pkl'), 'rb') as f:
        median_global = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'name_freq_map.pkl'), 'rb') as f:
        name_freq_map = pickle.load(f)
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
    return median_brand, median_global, name_freq_map, preprocessor, model

median_brand, median_global, name_freq_map, preprocessor, model = load_model()


def extract_max_torque_rpm(x):
    if pd.isna(x):
        return np.nan
    return float(re.findall(r"\d+(?:\.\d+)?", str(x))[-1].replace(',', '.'))


def normalize_torque(x):
    if pd.isna(x):
        return np.nan
    
    value = float(re.findall(r"\d+(?:\.\d+)?", str(x))[0].replace(',', '.'))
    if 'kg' in str(x).lower():
        value *= 9.806652

    return value


def preprocessing_df(df_input):

    df = df_input.copy()

    # Удаление единиц измерения mileage, engine и max_power
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    # Обработка столбца torque
    if 'max_torque_rpm' not in df.columns:
        df['max_torque_rpm'] = df['torque'].apply(extract_max_torque_rpm)
    df['torque'] = df['torque'].apply(normalize_torque)
        
    # Флаги пропусков
    for col in ['mileage', 'engine', 'max_power', 'torque', 'seats']:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    # добавление производителя и спецификации
    df['brand'] = df['name'].str.split().str[0]
    df['model'] = df['name'].str.split().str[1]
    df['submodel'] = df['name'].str.split().str[2]
    df['sub_specific'] = df['name'].str.split().str[-2]
    df['specific'] = df['name'].str.split().str[-1]

    # Полный привод
    df['if_4wd'] = df['name'].apply(
        lambda x: '4wd' in x.lower().split() or '4x4' in x.lower().split() or 'awd' in x.lower().split())

    # Заполнение пропусков
    cols_to_fill = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
    for col in cols_to_fill:
        df[col] = df[col].fillna(df['brand'].map(median_brand[col])).fillna(median_global[col])

    # Добавления отношения / произведения признаков
    df['power_per_cc'] = df['max_power'] / df['engine']             # Лошадей на куб
    df['mileage_per_cc'] = df['mileage'] / df['engine']             # Расход на куб
    df['power_per_year'] = df['max_power'] / (df['year'] - 1960)    # Лошадей к году выпуска
    df['age'] = 2022 - df['year']                                   # возраст машины
    df['age_power_mul'] = df['age'] * df['max_power']               # произведение мощности на возраст
    
    # Добавление квадратов признаков
    for col in ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'age']:   # попробовать убрать engine
        df[f'{col}_squad'] = df[col].apply(lambda x: x**2)
    
    # Частотное кодирование столбца name
    df['name'] = df['name'].map(name_freq_map).fillna(0)

    # Применение пайплайна OHE+scaler
    X = df.drop(columns='selling_price', errors='ignore')
    X = preprocessor.transform(X)

    return X


def parse_original_feature(f):
    """Извлечение названий призноков"""
    if f.startswith("ohe__"):
        clean = f.replace("ohe__", "")
        return clean.rsplit("_", 1)[0]
    if f.startswith("scaler__"):
        return f.replace("scaler__", "")
    return f


def group_feature_names(feature_names):
    """Группировка OHE"""
    groups = {}
    for f in feature_names:
        base = parse_original_feature(f)
        groups.setdefault(base, []).append(f)
    return groups


def aggregate_feature_weights(feature_names, weights):
    """Сумма абсолютных весов по группам + полный список."""
    df_full = pd.DataFrame({"Feature": feature_names, "Weight": weights})
    groups = group_feature_names(feature_names)

    aggregated = []
    for base, cols in groups.items():
        total = df_full[df_full["feature"].isin(cols)]["weight"].abs().sum()
        aggregated.append((base, total))

    df_grouped = (
        pd.DataFrame(aggregated, columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
    )
    return df_full, df_grouped


tabs = st.tabs(["Preds", "EDA", "Weights"])

with tabs[0]:

    st.header("Оценка автомобиля по введенным параметрам")

    with st.form("params_auto"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Car name (model)", "Maruti Swift Dzire VDI")
            year = st.number_input("Year", min_value=1961, max_value=2025, value=2014)
            km_driven = st.number_input("Kilometers driven", min_value=0, max_value=2_000_000, value=50_000)
            fuel = st.selectbox("Fuel type", ["Diesel", "Petrol", "CNG", "LPG"])
            seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.selectbox(
                "Owner",
                [
                    "First Owner", "Second Owner", "Third Owner",
                    "Fourth & Above Owner", "Test Drive Car"
                ]
            )

        with col2:
            mileage = st.number_input("Mileage (kmpl)", min_value=3.0, max_value=50.0, value=10.0)
            engine = st.number_input("Engine (CC)", min_value=500, max_value=3_000, value=1_200)
            max_power = st.number_input("Power (bhp)", min_value=30.0, max_value=1200.0, value=90.0)
            torque = st.number_input("Torque (Nm)", min_value=20.0, max_value=500.0, value=100.0)
            max_torque_rpm = st.number_input("Max torque RPM", min_value=1000, max_value=20_000, value=4000)
            seats = st.number_input("Seats", min_value=1, max_value=15, value=5)
            
        submitted = st.form_submit_button("Рассчитать цену")

    if submitted:
        # Датафрейм
        df = pd.DataFrame([{
            "name": name,
            "year": year,
            "km_driven": km_driven,
            "engine": engine,
            "max_power": max_power,
            "torque": torque,
            "max_torque_rpm": max_torque_rpm,
            "mileage": mileage,
            "seats": seats,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner
        }])
        
        try:
            X = preprocessing_df(df)
            preds = model.predict(X)[0]

            st.success(f"Цена автомобиля: **{preds:,.0f}**")

        except Exception as e:
            st.error(f"Ошибка при расчете цены: {e}")


    st.header("Оценка автомобиля по параметрам из CSV")
    st.write("Обязательные параметры: name, year, km_driven, fuel, seller_type, transmission, owner")

    uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Проверка обязательных колонок
            required_cols = ["name", "year", "km_driven", "fuel",
                            "seller_type", "transmission", "owner"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                st.error(f"Отсутствуют обязательные колонки: {missing}")

            else:
                # Если нет необязательных колоное - дополним
                non_required_cols = ['mileage', 'engine', 'max_power', 'torque', 'seats']
                for col in non_required_cols:
                    if col not in df.columns:
                        df[col] = np.nan_to_num
                
                # Препроцессинг
                X = preprocessing_df(df)

                # Предикт
                preds = model.predict(X)
                df["price_preds"] = preds
                st.subheader("Результат предсказания")
                st.dataframe(df)

                # Расчет метрик, если столбец с ценой есть
                if "selling_price" in df.columns:
                    y_true = df["selling_price"].values
                    mse = mean_squared_error(y_true, preds)
                    r2 = r2_score(y_true, preds)

                    st.subheader("Метрики:")
                    st.write(f"**MSE:** {mse:,.0f}")
                    st.write(f"**R2: ** {r2:,.4f}")

                # Возможность сохранения в CSV
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Скачать предикт в CSV",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")


with tabs[1]:

    st.header("Анализ обучающей выборки")

    #train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')

    # Минимально предобработаем
    for i in range(0, len(train)):
        try:
            ffil_value = float(train.loc[i, 'max_power'][:-4])
            train.loc[i, 'max_power'] = ffil_value

        except:
            if train.loc[i, 'max_power'] in [np.nan, 'nan']:
                continue
            elif train.loc[i, 'max_power'] == '0':
                ffil_value = float(train.loc[i, 'max_power'])
                train.loc[i, 'max_power'] = ffil_value
            else:
                train.loc[i, 'max_power'] = 0
    
    train.drop_duplicates(
        subset=[col for col in train.columns if col != 'selling_price'],
        keep='first',
        inplace=True
        )
    
    train.reset_index(drop=True, inplace=True)

    for cat_feature in ['mileage', 'engine', 'max_power']:
        train[cat_feature] = train[cat_feature].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    train['max_torque_rpm'] = train['torque'].apply(extract_max_torque_rpm)
    train['torque'] = train['torque'].apply(normalize_torque)

    train['brand'] = train['name'].str.split().str[0]
    cols_to_fill = ['mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm', 'seats']
    for col in cols_to_fill:
        train[col] = train[col].fillna(train['brand'].map(median_brand[col])).fillna(median_global[col])
    del train['brand']

    for col in ['engine', 'seats']:
        train[col] = train[col].astype(int)


    st.dataframe(train.head())

    num_cols = train.select_dtypes(include=["int", "float"]).columns

    # Попарные распределения числовых признаков
    st.subheader("Попарные распределения числовых признаков")
    fig = sns.pairplot(train[num_cols], diag_kind="kde")
    st.pyplot(fig)

    # Матрица корреляции Phik
    st.subheader("Матрица корреляции Phik")
    ph = train.phik_matrix()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(ph, cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Доп графики для категориальных фичей
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Цена по виду топлива")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(data=train, x="fuel", y="selling_price", palette='Set1', ax=ax)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, use_container_width=False)

    with col2:
        st.subheader("Цена по коробке передач")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.violinplot(data=train, x="transmission", y="selling_price", palette='Set1', ax=ax)
        st.pyplot(fig, use_container_width=False)

    with col3:
        st.subheader("Цена по типу продавца")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxenplot(data=train, x="seller_type", y="selling_price", palette='Set1', ax=ax)
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, use_container_width=False)


    with tabs[2]:

        try:
            feature_names = preprocessor.get_feature_names_out()
            weights = model.coef_.flatten()
            df_full, df_grouped = aggregate_feature_weights(feature_names, weights)

            st.header("Важность признаков")

            fig, ax = plt.subplots(figsize=(7, 8))
            sns.barplot(
                data=df_grouped,
                x="importance",
                y="feature",
                ax=ax
            )
            ax.set_title("Feature importances (по весам)")
            st.pyplot(fig)

            st.header("Веса модели")
            df_full_sorted = df_full.reindex(df_full["weight"].abs().sort_values(ascending=False).index)
            st.dataframe(df_full_sorted)

        except Exception as e:
            st.error(f"Ошибка при обработке весов модели: {e}")