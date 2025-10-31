"""
Скрипт для загрузки обученных моделей и генерации прогнозов на новые даты. 
Этот скрипт будет вызываться для получения актуальных прогнозов.
Ключевое отличие от model_trainer.py: здесь НЕТ пересчета квантилей и максимумов.
Все необходимые статистики (quantiles, max_vals) загружаются из сохраненных артефактов.
Это гарантирует полную согласованность между этапами.
"""
import pandas as pd
import numpy as np
import pickle
import os
from config import MODEL_DIR, HIGH_FREQUENCY_MODALITIES, SPECIALIZED_MODALITIES, RU_HOLIDAYS, PAY_WINDOW_START, PAY_WINDOW_END, PAY_WINDOW_MID_START, PAY_WINDOW_MID_END, FORECAST_START_DATE, FORECAST_PERIOD_DAYS
from model_trainer import add_calendar_features, final_postprocessing  # Импортируем функции для создания признаков и постобработки.

def load_model(filename):
    """Загружает модель (или артефакт) из файла."""
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Модель загружена: {filepath}")
    return model

def generate_future_dates(start_date, periods=60):
    """Генерирует DataFrame с будущими датами для прогноза."""
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    return pd.DataFrame({'ds': dates})

def predict_with_hybrid_model(modality, future_df):
    """
    Генерирует прогноз с помощью гибридной модели (Prophet + XGBoost).
    Для построения адаптивных интервалов и постобработки загружает исторические тренировочные данные.
    """
    # Загрузка моделей.
    prophet_model = load_model(f"{modality}_prophet.pkl")
    xgb_model = load_model(f"{modality}_xgboost.pkl")

    # Подготовка данных для Prophet: добавляем календарные признаки.
    future_df_with_feats = add_calendar_features(future_df)
    prophet_regressors = [
        'is_holiday', 'is_weekend', 'is_extended_holiday',
        'near_holiday', 'sin_dow', 'is_pay_window', 'sin_month'
    ]
    future_for_prophet = future_df_with_feats[['ds'] + [r for r in prophet_regressors if r in future_df_with_feats.columns]]

    # Прогноз Prophet.
    forecast = prophet_model.predict(future_for_prophet)
    forecast['yhat'] = np.clip(forecast['yhat'], 0, None)

    # АДАПТИВНАЯ ОБРЕЗКА ИНТЕРВАЛОВ
    # Загружаем тренировочные данные для расчета физических максимумов.
    from data_preprocessor import load_processed_data
    service_time_series = load_processed_data()
    train_df_raw = service_time_series[modality].reset_index()
    train_df_raw.columns = ['ds', 'y']
    train_df_raw['ds'] = pd.to_datetime(train_df_raw['ds'])
    train_df_raw['y'] = pd.to_numeric(train_df_raw['y'], errors='coerce').clip(lower=0)
    train_df_raw = train_df_raw.dropna(subset=['y']).reset_index(drop=True)
    # Добавляем календарные признаки.
    train_df = add_calendar_features(train_df_raw.copy())

    # Рассчитываем разные максимумы для рабочих дней и выходных/праздников.
    is_holiday_weekend_mask = (train_df['is_holiday'] == 1) | (train_df['is_weekend'] == 1)
    max_possible_weekday = train_df[~is_holiday_weekend_mask]['y'].quantile(0.999)
    max_possible_weekend = train_df[is_holiday_weekend_mask]['y'].quantile(0.999) if is_holiday_weekend_mask.sum() > 0 else max_possible_weekday

    # Добавляем признаки к forecast для определения типа дня.
    forecast_with_feats = add_calendar_features(forecast[['ds']].copy())
    is_forecast_weekend = (forecast_with_feats['is_holiday'] == 1) | (forecast_with_feats['is_weekend'] == 1)
    # Создаем вектор для максимально допустимого значения для каждого дня.
    max_vals = np.where(is_forecast_weekend, max_possible_weekend, max_possible_weekday)

    # ПОЛНОСТЬЮ СИММЕТРИЧНЫЙ И НЕОГРАНИЧЕННЫЙ ИНТЕРВАЛ
    # Рассчитываем ширину нижней части интервала
    lower_width = forecast['yhat'].values - forecast['yhat_lower'].values
    # Рассчитываем верхнюю границу как зеркально симметричную относительно yhat
    forecast['yhat_upper'] = forecast['yhat'].values + lower_width
    # Обрезаем ТОЛЬКО нижнюю границу до нуля (верхняя граница может быть любой)
    forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, None)

    # Извлекаем точечный прогноз Prophet.
    prophet_predictions = forecast.set_index('ds')['yhat']

    # Прогноз остатков с помощью XGBoost.
    xgb_features = [col for col in future_df_with_feats.columns if col not in ['ds', 'y']]
    X_future = future_df_with_feats.set_index('ds')[xgb_features].fillna(0)
    resid_forecast = xgb_model.predict(X_future)

    # Финальный прогноз.
    final_forecast = prophet_predictions.values + resid_forecast

    # Используем train_df (с признаками) для постобработки.
    final_forecast = final_postprocessing(final_forecast, prophet_predictions.values, future_df['ds'].values, modality, train_df)
    final_forecast = np.clip(final_forecast, 0, None)

    # Интервальный прогноз от Prophet (уже обрезанный!).
    yhat_lower = forecast.set_index('ds')['yhat_lower']
    yhat_upper = forecast.set_index('ds')['yhat_upper']

    # Формируем итоговый DataFrame.
    result_df = future_df.copy()
    result_df['y_pred'] = final_forecast  # <-- Исправлено
    result_df['yhat_lower'] = yhat_lower.values  # <-- Исправлено
    result_df['yhat_upper'] = yhat_upper.values  # <-- Исправлено

    return result_df

def predict_with_spike_model(modality, future_df):
    """
    Генерирует прогноз с помощью модели детекции всплесков.
    Ключевое улучшение: загружает ВСЕ необходимые артефакты (включая quantiles и max_vals) из одного файла.
    Это исключает необходимость пересчета статистик и предотвращает ошибки согласованности.
    """
    # Загружаем единый артефакт модели, содержащий ВСЕ необходимые компоненты.
    spike_model_artifacts = load_model(f"{modality}_spike_model.pkl")
    classifier_model = spike_model_artifacts['classifier']
    regressor_normal = spike_model_artifacts['regressor_normal']
    regressor_spike = regressor_normal  # Для упрощения (regressor_spike часто не используется отдельно).
    quantiles = spike_model_artifacts['quantiles']  # Загружаем предрассчитанные квантили ошибок.
    max_vals = spike_model_artifacts['max_vals']    # Загружаем предрассчитанные физические максимумы.
    
    print(f"✅ Загружены артефакты модели для {modality}: классификатор, регрессор, квантили, максимумы.")

    # Подготовка признаков для будущих дат.
    future_df_with_feats = add_calendar_features(future_df)
    feature_cols = [col for col in future_df_with_feats.columns if col not in ['ds', 'y', 'is_spike', 'month', 'dow', 'group_key']]
    X_future = future_df_with_feats[feature_cols]

    # Классификация: будет ли всплеск в этот день?
    spike_predictions = classifier_model.predict(X_future)

    # Генерация точечного прогноза.
    final_predictions = []
    for i, is_spike in enumerate(spike_predictions):
        if is_spike:
            pred = regressor_spike.predict(X_future.iloc[[i]])[0]
        else:
            pred = regressor_normal.predict(X_future.iloc[[i]])[0]
        final_predictions.append(max(0, pred))
    final_predictions = np.array(final_predictions)

    # ГЕНЕРАЦИЯ ИНТЕРВАЛОВ ДЛЯ БУДУЩЕГО (ИСПОЛЬЗУЕМ ЗАГРУЖЕННЫЕ quantiles и max_vals)
    yhat_lower_list = []
    yhat_upper_list = []
    for i, is_spike in enumerate(spike_predictions):
        pred = final_predictions[i]
        # Определяем тип дня для этого прогноза (рабочий или выходной/праздник).
        is_workday = future_df_with_feats.iloc[i]['is_workday']
        if is_workday:
            q_lower = quantiles['workday']['lower']
            q_upper = quantiles['workday']['upper']
            max_val = max_vals['workday']
        else:
            q_lower = quantiles['weekend_or_holiday']['lower']
            q_upper = quantiles['weekend_or_holiday']['upper']
            max_val = max_vals['weekend_or_holiday']

        # ВАЖНО: ВЫЧИСЛЯЕМ ГРАНИЦЫ И ДОБАВЛЯЕМ ИХ В СПИСКИ
        lower_bound = max(0, pred + q_lower)
        upper_bound = pred + q_upper
        # Обрезаем до физического максимума.
        lower_bound = min(lower_bound, max_val)
        upper_bound = min(upper_bound, max_val)
        yhat_lower_list.append(lower_bound)
        yhat_upper_list.append(upper_bound)

    # Формируем итоговый DataFrame.
    result_df = future_df.copy()
    result_df['y_pred'] = final_predictions
    result_df['yhat_lower'] = yhat_lower_list
    result_df['yhat_upper'] = yhat_upper_list

    return result_df

def generate_forecasts_for_all_modalities(start_date="2025-09-01", periods=60):
    """
    Генерирует прогнозы для всех модальностей на заданный период.
    Результаты сохраняются в существующий Excel-файл, добавляя новые листы с префиксом "Будущее_".
    """
    from config import RESULTS_DIR, FORECAST_START_DATE, FORECAST_PERIOD_DAYS
    future_df = generate_future_dates(FORECAST_START_DATE, FORECAST_PERIOD_DAYS)
    forecasts = {}

    for modality in HIGH_FREQUENCY_MODALITIES + SPECIALIZED_MODALITIES:
        print(f"\nГенерация прогноза для {modality}...")
        try:
            if modality in HIGH_FREQUENCY_MODALITIES:
                forecast_df = predict_with_hybrid_model(modality, future_df)
            elif modality in SPECIALIZED_MODALITIES:
                forecast_df = predict_with_spike_model(modality, future_df)
            else:
                print(f"Пропуск неизвестной модальности: {modality}")
                continue
            forecasts[modality] = forecast_df
            print(f"✅ Прогноз для {modality} сгенерирован. Размер: {forecast_df.shape}")
        except Exception as e:
            print(f"❌ Ошибка при генерации прогноза для {modality}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Сохранение прогнозов на будущее в Excel
    output_excel_path = os.path.join(RESULTS_DIR, "forecasting_results.xlsx")
    # Открываем существующий файл (если он есть) или создаем новый.
    mode = 'a' if os.path.exists(output_excel_path) else 'w'
    if mode == 'a':
        # Если файл существует, получаем список существующих листов.
        with pd.ExcelFile(output_excel_path) as xls:
            existing_sheets = xls.sheet_names
    else:
        existing_sheets = []

    with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
        # Если файл уже существовал, копируем старые листы (сводную таблицу и тестовые прогнозы).
        if mode == 'a':
            with pd.ExcelFile(output_excel_path) as xls:
                for sheet_name in existing_sheets:
                    if not sheet_name.startswith('Будущее_'):
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
        # Сохраняем прогнозы на будущее на отдельные листы.
        for modality, forecast_df in forecasts.items():
            if forecast_df is not None:
                sheet_name = f"Будущее_{modality}"[:31]  # Ограничение Excel.
                forecast_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅✅✅ Прогнозы на будущее сохранены в файл: {output_excel_path} ✅✅✅")

    return forecasts

if __name__ == "__main__":
    # Генерируем прогноз на 123 дня вперед (значение из config.py).
    all_forecasts = generate_forecasts_for_all_modalities()
    print("\n🎉 Генерация прогнозов завершена.")