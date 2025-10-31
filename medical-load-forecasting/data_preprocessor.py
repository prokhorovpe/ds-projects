"""
Скрипт для предобработки данных и создания временных рядов service_time_series. 
Основная задача: преобразовать сырые данные (дата + ID исследования + тип) в агрегированные временные ряды (дата + количество исследований).
Результат сохраняется в файл для последующего использования в обучении и прогнозировании.
"""
import pandas as pd
import numpy as np
import pickle
from config import START_DATE, END_DATE, PROCESSED_DATA_PATH, SERVICES_DICT
from data_loader import load_raw_data_chunked

def create_modality_time_series(df_cleaned):
    """
    Создает временные ряды по каждой модальности.
    Для каждой модальности:
      1. Фильтрует данные по типу услуги.
      2. Группирует по дате и считает количество исследований.
      3. Создает полный временной ряд от START_DATE до END_DATE, заполняя нулями дни без исследований.
    Это гарантирует, что все временные ряды имеют одинаковую длину и структуру.
    """
    start_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp(END_DATE)
    print(f"Создание временных рядов для периода: {start_date.date()} - {end_date.date()}")
    service_time_series = {}

    # Проходим по всем уникальным типам услуг в данных.
    for service in df_cleaned['type_of_service'].unique():
        full_name = SERVICES_DICT.get(service, service)  # Получаем полное название для логов.
        print(f"Обработка модальности: {service} ({full_name})")

        # Фильтруем данные по текущей модальности.
        df_service = df_cleaned[df_cleaned['type_of_service'] == service].copy()
        df_service['study_date'] = pd.to_datetime(df_service['study_date'], errors='coerce')
        df_service = df_service.dropna(subset=['study_date'])  # Удаляем записи с некорректными датами.
        # Фильтруем по заданному периоду.
        df_service = df_service[
            (df_service['study_date'] >= start_date) & 
            (df_service['study_date'] <= end_date)
        ]

        if len(df_service) == 0:
            print(f"  ⚠️ Нет данных для модальности {service}")
            continue

        # Агрегация: группировка по дате и подсчет количества исследований.
        time_series = (
            df_service
            .groupby('study_date')['inventory_number']
            .count()
            .reset_index(name='total_studies')  # Переименовываем столбец с подсчетом.
            .set_index('study_date')
            .sort_index()
            # Создаем полный диапазон дат и заполняем пропущенные дни нулями.
            .reindex(pd.date_range(start=start_date, end=end_date), fill_value=0)
            .astype(int)  # Приводим к целому типу.
        )
        service_time_series[service] = time_series
        print(f"  ✅ Временной ряд создан. Размер: {len(time_series)} дней. Всего исследований: {time_series['total_studies'].sum()}")

    return service_time_series

def save_processed_data(service_time_series):
    """Сохраняет обработанные временные ряды в файл в формате pickle."""
    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump(service_time_series, f)
    print(f"Обработанные данные сохранены в {PROCESSED_DATA_PATH}")

def load_processed_data():
    """Загружает обработанные временные ряды из файла. Используется в model_trainer.py и predictor.py."""
    with open(PROCESSED_DATA_PATH, 'rb') as f:
        service_time_series = pickle.load(f)
    print(f"Обработанные данные загружены из {PROCESSED_DATA_PATH}")
    return service_time_series

def main():
    """Основная функция для запуска предобработки."""
    df = load_raw_data_chunked()  # Загружаем сырые данные.
    service_time_series = create_modality_time_series(df) # Создаем временные ряды.
    save_processed_data(service_time_series) # Сохраняем результат.
    return service_time_series

if __name__ == "__main__":
    main()