"""
Главный скрипт для запуска всего пайплайна: от загрузки данных до генерации финального прогноза.
Предназначен для удобного запуска всех этапов последовательно.
"""
from data_preprocessor import main as run_preprocessing
from model_trainer import train_and_evaluate_models
from predictor import generate_forecasts_for_all_modalities
import os

def main():
    """Основная функция, запускающая все этапы пайплайна."""
    print("🚀 Запуск полного пайплайна прогнозирования...")

    # Этап 1: Предобработка данных.
    print("\n--- Этап 1: Предобработка данных ---")
    run_preprocessing()

    # Этап 2: Обучение и оценка моделей.
    print("\n--- Этап 2: Обучение моделей ---")
    train_and_evaluate_models()

    # Этап 3: Генерация финального прогноза.
    print("\n--- Этап 3: Генерация финального прогноза ---")
    generate_forecasts_for_all_modalities()

    print("\n✅ Пайплайн успешно завершен!")
    from config import RESULTS_DIR
    final_report_path = os.path.join(RESULTS_DIR, "forecasting_results.xlsx")
    print(f"\n📊📊📊 ВСЕ РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {final_report_path} 📊📊📊")

if __name__ == "__main__":
    main()