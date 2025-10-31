"""
Скрипт для загрузки исходных данных. Загружает только необходимые столбцы, чтобы экономить память.
Основная задача: получить сырые данные из CSV и удалить дубликаты по уникальному идентификатору.
"""
import pandas as pd
import os
import subprocess
from config import RAW_DATA_PATH, REQUIRED_COLUMNS

def download_data_if_needed():
    """
    Скачивает файл с данными из Google Drive, если его еще нет локально.
    Для работы требуется установленная утилита 'gdown'. Если ее нет, скрипт выдаст ошибку.
    """
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Файл {RAW_DATA_PATH} не найден. Начинаю загрузку...")
        download_url = "https://drive.google.com/uc?id=1OFA2de_VSwm-OxuTjDvshETCT940jeSG"
        try:
            subprocess.run(["gdown", download_url, "-O", RAW_DATA_PATH], check=True)
            print(f"Данные успешно загружены в {RAW_DATA_PATH}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Ошибка при скачивании файла: {e}")
        except FileNotFoundError:
            raise Exception("Утилита 'gdown' не найдена. Установите ее: pip install gdown")
    else:
        print(f"Файл данных уже существует: {RAW_DATA_PATH}")

def load_raw_data_chunked(chunksize=500_000):
    """
    Загружает огромный CSV-файл по частям (чанкам), чтобы не перегружать оперативную память.
    Одновременно удаляет дубликаты по столбцу 'inventory_number' на лету.
    Это критически важно, так как дубликаты могут серьезно исказить прогноз.
    """
    download_data_if_needed()
    print("Загрузка сырых данных по частям...")
    
    # Используем set() для отслеживания уже загруженных уникальных ID.
    unique_inventory = set()
    collected_data = []

    # Читаем файл чанками. low_memory=False отключает предупреждения о смешанных типах данных.
    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=REQUIRED_COLUMNS, chunksize=chunksize, low_memory=False):
        # Удаляем дубликаты внутри текущего чанка.
        chunk = chunk.drop_duplicates(subset=['inventory_number'])
        # Фильтруем записи, которые уже были загружены в предыдущих чанках.
        chunk = chunk[~chunk['inventory_number'].isin(unique_inventory)]
        # Обновляем сет уникальных ID.
        unique_inventory.update(chunk['inventory_number'].tolist())
        # Сохраняем чанк для последующего объединения.
        collected_data.append(chunk)
        print(f"  Загружено {len(unique_inventory):,} уникальных записей...")

    # Объединяем все чанки в один итоговый DataFrame.
    df = pd.concat(collected_data, ignore_index=True)
    print(f"✅ Данные полностью загружены и очищены от дубликатов. Итоговый размер: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_raw_data_chunked()
    print(df.head()) 