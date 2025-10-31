"""
Скрипт для обучения моделей машинного обучения на исторических данных.
Для каждой модальности выбирается и обучается подходящая модель:
- Для HIGH_FREQUENCY_MODALITIES: Гибридная модель Prophet + XGBoost для остатков.
- Для SPECIALIZED_MODALITIES: Модель детекции всплесков (XGBoost Classifier + 2 XGBoost Regressor).
Результаты обучения (модели, метрики) сохраняются на диск.
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Игнорируем предупреждения от Prophet и sklearn.
from prophet import Prophet
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from config import (
    MODEL_DIR, RESULTS_DIR, TEST_START_DATE, HIGH_FREQUENCY_MODALITIES, SPECIALIZED_MODALITIES,
    PAY_WINDOW_START, PAY_WINDOW_END, PAY_WINDOW_MID_START, PAY_WINDOW_MID_END, RU_HOLIDAYS, RANDOM_STATE
)
from data_preprocessor import load_processed_data

def symmetric_mape(y_true, y_pred):
    """Рассчитывает симметричную MAPE (sMAPE), которая более устойчива к нулевым значениям."""
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

# Полный набор календарных признаков
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к DataFrame с датами (столбец 'ds') множество календарных признаков.
    Это ключевая функция, используемая как при обучении, так и при прогнозировании.
    Согласованность признаков между этапами обучения и прогнозирования КРИТИЧЕСКИ важна.
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    # Базовые признаки даты.
    df['month'] = df['ds'].dt.month.astype('int8')
    df['day_of_month'] = df['ds'].dt.day.astype('int8')
    df['week_of_year'] = df['ds'].dt.isocalendar().week.astype('int8')
    df['quarter'] = df['ds'].dt.quarter.astype('int8')
    df['year'] = df['ds'].dt.year.astype('int16')
    df['dow'] = df['ds'].dt.weekday.astype('int8')  # День недели (0=Пн, 6=Вс)
    
    # Циклические признаки для корректной обработки периодичности (например, понедельник ближе к воскресенью, чем к среде).
    df['sin_dow'] = np.sin(2 * np.pi * df['dow'] / 7).astype('float32')
    df['cos_dow'] = np.cos(2 * np.pi * df['dow'] / 7).astype('float32')
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12).astype('float32')
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['ds'].dt.dayofyear / 365).astype('float32')
    
    # Праздники и выходные.
    df['is_holiday'] = df['ds'].isin(RU_HOLIDAYS).astype('int8')
    df['is_weekend'] = (df['dow'] >= 5).astype('int8')  # Сб=5, Вс=6
    
    # Бизнес-логика: признаки "окон выплат" (начало и середина месяца).
    df['is_pay_window'] = (
        ((df['day_of_month'] >= PAY_WINDOW_START) & (df['day_of_month'] <= PAY_WINDOW_END)) |
        ((df['day_of_month'] >= PAY_WINDOW_MID_START) & (df['day_of_month'] <= PAY_WINDOW_MID_END))
    ).astype('int8')
    
    # Расширенные праздничные признаки: сколько дней до/после ближайшего праздника.
    df['days_to_holiday'] = df['ds'].apply(
        lambda x: min([(h - x).days for h in RU_HOLIDAYS if h > x] or [365])
    ).clip(0, 7).astype('int8')  # Ограничиваем до 7 дней.
    df['days_after_holiday'] = df['ds'].apply(
        lambda x: min([(x - h).days for h in RU_HOLIDAYS if h < x] or [365])
    ).clip(0, 7).astype('int8')
    df['near_holiday'] = ((df['days_to_holiday'] <= 2) | (df['days_after_holiday'] <= 2)).astype('int8')
    
    # Признак "длинных" праздников (если в радиусе ±2 дня от текущей даты есть 3+ праздника).
    def is_extended_holiday(date):
        count = sum(1 for i in range(-2, 3) if (date + pd.Timedelta(days=i)) in RU_HOLIDAYS)
        return 1 if count >= 3 else 0
    df['is_extended_holiday'] = df['ds'].apply(is_extended_holiday).astype('int8')
    
    # Дополнительные бизнес-признаки.
    df['is_month_start'] = (df['day_of_month'] == 1).astype('int8')
    df['is_month_end'] = (df['ds'].dt.is_month_end).astype('int8')
    df['is_quarter_start'] = ((df['month'] - 1) % 3 == 0) & (df['day_of_month'] <= 7).astype('int8')
    df['is_quarter_end'] = ((df['month'] % 3 == 0) & (df['day_of_month'] >= 24)).astype('int8')
    
    # Рабочий день = не выходной и не праздник.
    df['is_workday'] = (~(df['is_weekend'] | df['is_holiday'])).astype('int8')
    
    return df

# --- Функции создания моделей ---

def create_prophet_model(modality: str):
    """
    Создает и настраивает модель Prophet для заданной модальности.
    Параметры подбираются индивидуально для каждой модальности и типа модели.
    """
    if modality in SPECIALIZED_MODALITIES:
        # Базовые параметры для низкочастотных модальностей.
        base_params = {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'mcmc_samples': 0,
            'changepoint_prior_scale': 0.001, # Очень низкая чувствительность к изменениям тренда.
            'growth': 'linear'
        }
    else:
        # Базовые параметры для высокочастотных модальностей.
        base_params = {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative', # Мультипликативная сезонность лучше для данных с растущей волатильностью.
            'mcmc_samples': 0,
            'growth': 'linear'
        }

    # Индивидуальные параметры для каждой модальности.
    modality_params = {
    'РГ': {'changepoint_prior_scale': 0.03, 'seasonality_prior_scale': 10.0},
    'ММГ': {'changepoint_prior_scale': 0.03, 'seasonality_prior_scale': 8.0},
    'ФЛГ': {'seasonality_prior_scale': 12.0},
    'КТ': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 6.0},
    'МРТ': {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 5.0},
    'Денс': {'seasonality_prior_scale': 4.0}
    }

    base_params.update(modality_params.get(modality, {}))
    return Prophet(**base_params)

def create_xgboost_model_for_residuals(modality: str):
    """
    Создает модель XGBoost для прогнозирования остатков после Prophet.
    Параметры регуляризации (reg_alpha, reg_lambda) подобраны для предотвращения переобучения.
    """
    if modality in SPECIALIZED_MODALITIES:
        params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'reg_alpha': 2.0,  # L1 регуляризация
            'reg_lambda': 10.0, # L2 регуляризация
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    else:
        params = {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.1,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        }
    return XGBRegressor(**params)

# ФУНКЦИЯ ДЕТЕКЦИИ ВСПЛЕСКОВ
def create_classification_regression_model(modality, train_df, test_df):
    """
    Создает и обучает модель детекции всплесков для низкочастотных модальностей.
    Алгоритм:
      1. На тренировочных данных выявляются "всплески" (аномально высокие значения) с помощью Z-оценки.
      2. Обучается классификатор (XGBoost) для предсказания, будет ли всплеск в заданный день.
      3. Обучаются две модели регрессии: для "нормальных" дней и для дней со "всплесками".
      4. На тесте: сначала классификатор определяет тип дня, затем соответствующий регрессор делает прогноз.
    """
    train_df = train_df.copy()
    # Создаем признаки для группировки: месяц + день недели.
    train_df['month'] = train_df['ds'].dt.month
    train_df['dow'] = train_df['ds'].dt.weekday
    train_df['group_key'] = train_df['month'].astype(str) + '_' + train_df['dow'].astype(str)
    train_df = train_df.sort_values('ds').reset_index(drop=True)
    train_df['is_spike'] = 0

    # Выявление всплесков: рассчитываем скользящее среднее и стандартное отклонение для каждой группы.
    for group_key in train_df['group_key'].unique():
        mask = train_df['group_key'] == group_key
        if mask.sum() < 5:  # Пропускаем группы с малым количеством данных.
            continue
        group_data = train_df[mask].copy()
        rolling_mean = group_data['y'].rolling(window=30, min_periods=5).mean()
        rolling_std = group_data['y'].rolling(window=30, min_periods=5).std()
        # Заполняем NaN для начала ряда расширяющимся средним.
        rolling_mean = rolling_mean.fillna(group_data['y'].expanding().mean())
        rolling_std = rolling_std.fillna(group_data['y'].expanding().std())
        # Рассчитываем Z-оценку.
        z_score = (group_data['y'] - rolling_mean) / (rolling_std + 1e-8)
        # Сдвигаем на 1, чтобы использовать предыдущее значение для предсказания текущего.
        z_score_shifted = z_score.shift(1).fillna(False).infer_objects(copy=False)
        # Маска всплесков: Z-оценка > 1.5.
        spike_mask = (z_score_shifted > 1.5)
        train_df.loc[mask, 'is_spike'] = spike_mask.astype(int)

    # Fallback с понижением порога
    # Если всплесков не найдено, пробуем более простой метод: значение > 1.5 * скользящее среднее.
    if train_df['is_spike'].sum() == 0:
        print(f"⚠️ Для {modality}: Всплесков не обнаружено. Пробуем понизить порог...")
        for group_key in train_df['group_key'].unique():
            mask = train_df['group_key'] == group_key
            if mask.sum() < 5:
                continue
            group_data = train_df[mask].copy()
            rolling_mean = group_data['y'].rolling(window=30, min_periods=5).mean()
            rolling_mean = rolling_mean.fillna(group_data['y'].expanding().mean())
            spike_mask = (group_data['y'] > rolling_mean * 1.5)
            spike_mask = spike_mask.shift(1).fillna(False).infer_objects(copy=False)
            train_df.loc[mask, 'is_spike'] = spike_mask.astype(int)
        # Если и после этого всплесков нет, используем простую регрессию.
        if train_df['is_spike'].sum() == 0:
            print(f"⚠️ Для {modality}: Данные очень гладкие. Используется упрощенная регрессия.")
            train_df_with_feats = add_calendar_features(train_df)
            test_df_with_feats = add_calendar_features(test_df)
            feature_cols = [col for col in train_df_with_feats.columns if col not in ['ds', 'y', 'is_spike', 'month', 'dow', 'group_key']]
            model = XGBRegressor(n_estimators=150, max_depth=4, random_state=RANDOM_STATE)
            X_train = train_df_with_feats[feature_cols]
            y_train = train_df_with_feats['y']
            model.fit(X_train, y_train)
            X_test = test_df_with_feats[feature_cols]
            final_predictions = model.predict(X_test)
            final_predictions = np.clip(final_predictions, 0, None)
            # ПОСТОБРАБОТКА В FALLBACK
            final_predictions = final_postprocessing(final_predictions, np.zeros_like(final_predictions), test_df['ds'].values, modality, train_df)
            # Возвращаем только 4 значения, так как классификатор и второй регрессор не используются.
            return final_predictions, model, None, None

    # Создаем модели.
    models = {
        'classifier': XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
        'regressor_normal': XGBRegressor(n_estimators=150, max_depth=4, random_state=RANDOM_STATE),
        'regressor_spike': XGBRegressor(n_estimators=150, max_depth=4, random_state=RANDOM_STATE)
    }

    # Добавляем календарные признаки.
    train_df_with_feats = add_calendar_features(train_df)
    feature_cols = [col for col in train_df_with_feats.columns if col not in ['ds', 'y', 'is_spike', 'month', 'dow', 'group_key']]

    # Обучаем классификатор.
    X_train_clf = train_df_with_feats[feature_cols]
    y_train_clf = train_df_with_feats['is_spike']
    models['classifier'].fit(X_train_clf, y_train_clf)

    # Обучаем регрессор для "нормальных" дней.
    normal_mask = train_df_with_feats['is_spike'] == 0
    X_train_normal = train_df_with_feats[normal_mask][feature_cols]
    y_train_normal = train_df_with_feats.loc[normal_mask, 'y']
    models['regressor_normal'].fit(X_train_normal, y_train_normal)

    # Обучаем регрессор для дней со "всплесками". Если данных мало, используем регрессор для нормальных дней.
    spike_mask = train_df_with_feats['is_spike'] == 1
    if spike_mask.sum() > 10:
        X_train_spike = train_df_with_feats[spike_mask][feature_cols]
        y_train_spike = train_df_with_feats.loc[spike_mask, 'y']
        models['regressor_spike'].fit(X_train_spike, y_train_spike)
    else:
        models['regressor_spike'] = models['regressor_normal']

    # Генерируем прогноз на тестовых данных.
    test_df_with_feats = add_calendar_features(test_df)
    X_test = test_df_with_feats[feature_cols]
    spike_predictions = models['classifier'].predict(X_test)

    final_predictions = []
    for i, is_spike in enumerate(spike_predictions):
        if is_spike:
            pred = models['regressor_spike'].predict(X_test.iloc[[i]])[0]
        else:
            pred = models['regressor_normal'].predict(X_test.iloc[[i]])[0]
        final_predictions.append(max(0, pred))

    # Рассчитываем ошибки на тренировке для построения доверительных интервалов.
    train_predictions = []
    for i in range(len(X_train_clf)):
        is_spike = y_train_clf.iloc[i]
        if is_spike:
            pred = models['regressor_spike'].predict(X_train_clf.iloc[[i]])[0]
        else:
            pred = models['regressor_normal'].predict(X_train_clf.iloc[[i]])[0]
        train_predictions.append(max(0, pred))

    train_predictions = np.array(train_predictions)
    train_actual = train_df_with_feats.loc[X_train_clf.index, 'y'].values
    train_errors = train_actual - train_predictions

    # Разделяем ошибки на нормальные и всплесковые.
    normal_errors = train_errors[y_train_clf == 0]
    spike_errors = train_errors[y_train_clf == 1] if (y_train_clf == 1).sum() > 10 else normal_errors

    # ФУНКЦИЯ ДЛЯ РАСЧЕТА КВАНТИЛЕЙ С УЧЕТОМ МАЛОГО ОБЪЕМА
    def safe_quantiles(errors, q_low=0.05, q_high=0.95):
        """
        Безопасно вычисляет квантили, даже если данных мало.
        Если < 5 точек — используем медиану ± 2 * MAD (Median Absolute Deviation).
        Это более устойчиво, чем квантили, на малых выборках.
        """
        if len(errors) < 5:
            median = np.median(errors)
            mad = np.median(np.abs(errors - median))
            return median - 2 * mad, median + 2 * mad
        else:
            return np.quantile(errors, q_low), np.quantile(errors, q_high)

    # Применяем безопасные квантили.
    normal_lower_quantile, normal_upper_quantile = safe_quantiles(normal_errors, 0.05, 0.95)
    spike_lower_quantile, spike_upper_quantile = safe_quantiles(spike_errors, 0.05, 0.95)

    # Формируем словарь квантилей. Важно: для рабочих дней - нормальные ошибки, для выходных - всплесковые.
    quantiles = {
        'workday': {'lower': normal_lower_quantile, 'upper': normal_upper_quantile},
        'weekend_or_holiday': {'lower': spike_lower_quantile, 'upper': spike_upper_quantile}
    }

    # Рассчитываем физические максимумы для каждого типа дня (99.9-й перцентиль).
    mask_workday = train_df_with_feats['is_workday'] == 1
    mask_weekend_or_holiday = (train_df_with_feats['is_weekend'] == 1) | (train_df_with_feats['is_holiday'] == 1)
    max_possible_workday = train_df_with_feats.loc[mask_workday, 'y'].quantile(0.999)
    max_possible_weekend_or_holiday = train_df_with_feats.loc[mask_weekend_or_holiday, 'y'].quantile(0.999)
    max_vals = {
        'workday': max_possible_workday,
        'weekend_or_holiday': max_possible_weekend_or_holiday
    }

    # ВОЗВРАЩАЕМ ТОЛЬКО 6 ЗНАЧЕНИЙ
    return (
        np.array(final_predictions),
        models['classifier'],
        models['regressor_normal'],
        models['regressor_spike'],
        quantiles,
        max_vals
    )

#  ФУНКЦИЯ АДАПТИВНОЙ ПОСТОБРАБОТКИ
def final_postprocessing(predictions, prophet_predictions, test_dates, modality, train_df):
    """
    Финальная постобработка прогноза. Ограничивает прогноз физически возможными значениями.
    1. Для выходных и праздничных дней: обрезает прогноз до 95-го перцентиля от исторических значений.
    2. Для дней, где Prophet предсказал почти ноль: обрезает прогноз до медианы низкой нагрузки.
    Это предотвращает завышенные или необоснованно низкие прогнозы.
    """
    test_df = pd.DataFrame({'ds': test_dates})
    test_features = add_calendar_features(test_df)
    # Маска для выходных и праздников.
    mask = (test_features['is_weekend'] == 1) | (test_features['is_holiday'] == 1)
    if mask.any():
        train_mask = (train_df['is_weekend'] == 1) | (train_df['is_holiday'] == 1)
        if train_mask.sum() > 10:
            max_val = train_df.loc[train_mask, 'y'].quantile(0.95)
        else:
            max_val = train_df['y'].median()
        predictions[mask] = np.minimum(predictions[mask], max_val)

    # Маска для дней с очень низким прогнозом от Prophet.
    low_threshold = train_df['y'].quantile(0.05) * 0.1
    zero_prophet_mask = prophet_predictions < low_threshold
    if zero_prophet_mask.any():
        low_load_mask = train_df['y'] < train_df['y'].quantile(0.1)
        if low_load_mask.sum() > 5:
            safe_median = train_df.loc[low_load_mask, 'y'].median()
        else:
            safe_median = train_df['y'].median() * 0.1
        predictions[zero_prophet_mask] = np.minimum(predictions[zero_prophet_mask], safe_median)

    return predictions

# --- Основные функции обучения ---

def run_hybrid_model_for_high_freq(train_df, test_df, modality):
    """
    Запускает гибридную модель (Prophet + XGBoost) для высокочастотных модальностей.
    """
    train_df = add_calendar_features(train_df)
    test_df = add_calendar_features(test_df)

    model = create_prophet_model(modality)
    # Определяем, какие регрессоры есть в данных и добавляем их в модель Prophet.
    prophet_regressors = [
        'is_holiday', 'is_weekend', 'is_extended_holiday', 
        'near_holiday', 'sin_dow', 'is_pay_window', 'sin_month'
    ]
    for regressor in prophet_regressors:
        if regressor in train_df.columns:
            model.add_regressor(regressor)

    # Обучаем Prophet на тренировочных данных.
    model.fit(train_df[['ds', 'y'] + prophet_regressors])
    future = test_df[['ds'] + prophet_regressors].copy()
    forecast = model.predict(future)

    # ОБРЕЗКА ВСЕХ КОМПОНЕНТОВ ПРОГНОЗА
    # Рассчитываем глобальный максимум для обрезки.
    max_possible_value = train_df['y'].quantile(0.999)
    # Обрезаем ВСЕ компоненты прогноза: yhat, yhat_lower, yhat_upper.
    forecast['yhat'] = np.clip(forecast['yhat'], 0, max_possible_value)
    forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, max_possible_value)
    forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], 0, max_possible_value)

    prophet_predictions = forecast.set_index('ds')['yhat']

    # Рассчитываем остатки на тренировке.
    train_forecast = model.predict(train_df[['ds'] + prophet_regressors])
    train_forecast['yhat'] = np.clip(train_forecast['yhat'], 0, None)
    train_residuals = train_df.set_index('ds')['y'] - train_forecast.set_index('ds')['yhat']

    # Обучаем XGBoost на остатках.
    xgb_model = create_xgboost_model_for_residuals(modality)
    xgb_features = [col for col in train_df.columns if col not in ['ds', 'y']]
    X_train = train_df.set_index('ds')[xgb_features].fillna(0)
    y_train = train_residuals.fillna(0)
    sample_weights = np.abs(y_train) + 1  # Веса пропорциональны абсолютному значению остатка.
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Прогноз остатков на тесте.
    X_test = test_df.set_index('ds')[xgb_features].fillna(0)
    resid_forecast = xgb_model.predict(X_test)

    # Финальный прогноз = прогноз Prophet + прогноз остатков.
    final_forecast = prophet_predictions.values + resid_forecast

    # АДАПТИВНАЯ ПОСТОБРАБОТКА
    final_forecast = final_postprocessing(final_forecast, prophet_predictions.values, test_df['ds'].values, modality, train_df)
    final_forecast = np.clip(final_forecast, 0, None)

    # ГЕНЕРАЦИЯ АДАПТИВНЫХ ИНТЕРВАЛОВ НА ТЕСТЕ
    # Рассчитываем ошибки гибридной модели на тренировке.
    train_final_forecast = train_forecast.set_index('ds')['yhat'].values + xgb_model.predict(X_train)
    train_errors = train_df['y'].values - train_final_forecast
    # Рассчитываем квантили ошибок.
    lower_quantile = np.quantile(train_errors, 0.05)
    upper_quantile = np.quantile(train_errors, 0.95)
    # Применяем квантили к точечному прогнозу на тесте.
    test_yhat = prophet_predictions.values
    test_resid = resid_forecast
    test_final_forecast = test_yhat + test_resid
    yhat_lower = test_final_forecast + lower_quantile
    yhat_upper = test_final_forecast + upper_quantile
    # Обрезаем до физических ограничений.
    max_possible_value = train_df['y'].quantile(0.999)
    yhat_lower = np.clip(yhat_lower, 0, max_possible_value)
    yhat_upper = np.clip(yhat_upper, 0, max_possible_value)
    # Формируем интервальный прогноз.
    interval_forecast = pd.DataFrame({
        'ds': test_df['ds'].values,
        'yhat_lower': yhat_lower,
        'yhat_upper': yhat_upper
    })


    return final_forecast, interval_forecast, model, xgb_model

def run_spike_detection_model_for_low_freq(train_df, test_df, modality):
    """
    Запускает специализированную модель (классификация всплесков + регрессия) для низкочастотных модальностей.
    """
    # Получаем квантили и максимумы, разбитые по типу дня
    final_predictions, classifier_model, regressor_normal, regressor_spike, quantiles, max_vals = create_classification_regression_model(modality, train_df, test_df)

    # Подготовка тестовых данных.
    test_df_with_feats = add_calendar_features(test_df)
    feature_cols = [col for col in test_df_with_feats.columns if col not in ['ds', 'y', 'is_spike', 'month', 'dow', 'group_key']]
    X_test = test_df_with_feats[feature_cols]
    spike_predictions = classifier_model.predict(X_test)

    # АДАПТИВНАЯ ГЕНЕРАЦИЯ ИНТЕРВАЛОВ ПО ТИПУ ДНЯ
    yhat_lower_list = []
    yhat_upper_list = []
    # Добавляем признаки типа дня к тестовым данным (дублируем логику из add_calendar_features для ясности).
    test_df_with_feats['is_weekend'] = (test_df_with_feats['dow'] >= 5).astype('int8')
    test_df_with_feats['is_holiday'] = test_df_with_feats['ds'].isin(RU_HOLIDAYS).astype('int8')
    test_df_with_feats['is_workday'] = ~(test_df_with_feats['is_weekend'] | test_df_with_feats['is_holiday'])

    for i, is_spike in enumerate(spike_predictions):
        pred = final_predictions[i]
        # Определяем тип дня для этого прогноза.
        is_workday = test_df_with_feats.iloc[i]['is_workday']
        if is_workday:
            q_lower = quantiles['workday']['lower']
            q_upper = quantiles['workday']['upper']
            max_val = max_vals['workday']
        else:
            q_lower = quantiles['weekend_or_holiday']['lower']
            q_upper = quantiles['weekend_or_holiday']['upper']
            max_val = max_vals['weekend_or_holiday']
        # Вычисляем интервал.
        lower_bound = max(0, pred + q_lower)
        upper_bound = pred + q_upper
        # Обрезаем до физического максимума.
        lower_bound = min(lower_bound, max_val)
        upper_bound = min(upper_bound, max_val)
        yhat_lower_list.append(lower_bound)
        yhat_upper_list.append(upper_bound)

    interval_forecast = pd.DataFrame({
        'ds': test_df['ds'].values,
        'yhat_lower': yhat_lower_list,
        'yhat_upper': yhat_upper_list
    })

    # Возвращаем 6 значений, включая quantiles и max_vals для сохранения.
    return final_predictions, interval_forecast, classifier_model, regressor_normal, quantiles, max_vals

def save_model(model, filename):
    """Сохраняет модель (или артефакт) в файл в формате pickle."""
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Модель сохранена: {filepath}")

def train_and_evaluate_models():
    """
    Основная функция. Загружает данные, разбивает на train/test, обучает модели для всех модальностей,
    рассчитывает метрики и сохраняет результаты в Excel-файл.
    """
    service_time_series = load_processed_data()
    test_start = pd.to_datetime(TEST_START_DATE)
    results = {}
    forecast_tables = {}

    for modality in service_time_series.keys():
        print(f"\n{'='*60}")
        print(f"🎯 Обучение модели для {modality}")
        print(f"{'='*60}")

        ts_data = service_time_series[modality]

        # Универсальная обработка данных: приведение к формату DataFrame с колонками ['ds', 'y'].
        if isinstance(ts_data, pd.Series):
            ts_data = ts_data.reset_index()
            ts_data.columns = ['ds', 'y']
        elif isinstance(ts_data, pd.DataFrame):
            if len(ts_data.columns) == 1:
                ts_data = ts_data.reset_index()
                ts_data.columns = ['ds', 'y']
            elif len(ts_data.columns) >= 2:
                ts_data = ts_data.iloc[:, :2].copy()
                ts_data.columns = ['ds', 'y']
            else:
                raise ValueError(f"DataFrame для модальности {modality} не содержит данных")
        else:
            raise TypeError(f"Неподдерживаемый тип данных для {modality}: {type(ts_data)}")

        ts_data['ds'] = pd.to_datetime(ts_data['ds'])
        ts_data['y'] = pd.to_numeric(ts_data['y'], errors='coerce').clip(lower=0)
        ts_data = ts_data.dropna(subset=['y']).reset_index(drop=True)

        # Разделение на train и test.
        train_df = ts_data[ts_data['ds'] < test_start].copy()
        test_df = ts_data[ts_data['ds'] >= test_start].copy()

        if len(test_df) == 0:
            print(f"⚠️ Нет данных для теста в модальности {modality}")
            continue

        try:
            if modality in HIGH_FREQUENCY_MODALITIES:
                final_predictions, interval_forecast, prophet_model, auxiliary_model = run_hybrid_model_for_high_freq(train_df, test_df, modality)
                save_model(prophet_model, f"{modality}_prophet.pkl")
                save_model(auxiliary_model, f"{modality}_xgboost.pkl")
                model_type = 'prophet_xgboost'
            elif modality in SPECIALIZED_MODALITIES:
                final_predictions, interval_forecast, classifier_model, regressor_model, quantiles, max_vals = run_spike_detection_model_for_low_freq(train_df, test_df, modality)
                # Создаем единый объект для сохранения ВСЕХ артефактов модели.
                spike_model_artifacts = {
                    'classifier': classifier_model,
                    'regressor_normal': regressor_model,
                    'quantiles': quantiles,
                    'max_vals': max_vals
                }
                save_model(spike_model_artifacts, f"{modality}_spike_model.pkl") # Сохраняем ВСЕ в один файл
                model_type = 'spike_detection'
            else:
                raise ValueError(f"Неизвестная модальность: {modality}")

            # Рассчитываем метрики качества на тесте.
            test_y = test_df['y'].values
            mape = symmetric_mape(test_y, final_predictions)
            r2 = r2_score(test_y, final_predictions)
            rmse = np.sqrt(mean_squared_error(test_y, final_predictions))
            mae = mean_absolute_error(test_y, final_predictions)

            results[modality] = {
                'modality': modality,
                'model_type': model_type,
                'final_metrics': {'MAPE': mape, 'R2': r2, 'RMSE': rmse, 'MAE': mae},
                'test_actual': test_y,
                'test_final_pred': final_predictions,
            }

            # Формируем таблицу с прогнозами на тесте.
            forecast_table = test_df[['ds']].copy()
            forecast_table['y_true'] = test_y
            forecast_table['y_pred'] = final_predictions
            forecast_table['yhat_lower'] = interval_forecast.set_index('ds').loc[forecast_table['ds'], 'yhat_lower'].values
            forecast_table['yhat_upper'] = interval_forecast.set_index('ds').loc[forecast_table['ds'], 'yhat_upper'].values
            forecast_tables[modality] = forecast_table

            print("📈 Метрики на тесте:")
            print(f"   MAPE: {mape:.3f}%")
            print(f"   R2: {r2:.4f}")
            print(f"   RMSE: {rmse:.1f}")
            print(f"   MAE: {mae:.1f}")

        except Exception as e:
            print(f"❌ Ошибка при обучении {modality}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[modality] = None
            forecast_tables[modality] = None

    # Вывод сводной таблицы результатов.
    print("\n" + "="*80)
    print("📋 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("="*80)
    summary_data = []
    for modality, result in results.items():
        if result:
            summary_data.append({
                'Модальность': modality,
                'MAPE': result['final_metrics']['MAPE'],
                'R2': result['final_metrics']['R2'],
                'RMSE': result['final_metrics']['RMSE'],
                'MAE': result['final_metrics']['MAE'],
                'Тип модели': result['model_type']
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    else:
        print("Нет результатов для отображения")
        summary_df = pd.DataFrame()

    # Сохранение ВСЕХ результатов в один Excel-файл
    output_excel_path = os.path.join(RESULTS_DIR, "forecasting_results.xlsx")
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # 1. Сводная таблица метрик.
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Сводная_таблица', index=False)
        # 2. Детальные прогнозы на тесте для каждой модальности.
        for modality, forecast_df in forecast_tables.items():
            if forecast_df is not None:
                sheet_name = f"Тест_{modality}"[:31]  # Excel ограничивает длину названия листа.
                forecast_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅✅✅ Все результаты обучения и прогнозы на тесте сохранены в файл: {output_excel_path} ✅✅✅")

    return results, forecast_tables

if __name__ == "__main__":
    results, forecast_tables = train_and_evaluate_models()
    print(f"\n🎉 Обучение завершено. Модели сохранены в папку '{MODEL_DIR}'")