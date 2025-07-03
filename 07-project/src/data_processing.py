# 07-project/src/data_processing.py (ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ)

import os
import pandas as pd

def process_downloaded_data():
    """
    Читает скачанные файлы, агрегирует данные по генам и объединяет
    в единую матрицу экспрессии, сохраняя ее в папку processed.
    """
    raw_data_path = "07-project/data/raw/"
    processed_data_path = "07-project/data/processed/"
    metadata_path = "07-project/data/metadata.csv"
    output_path = os.path.join(processed_data_path, "gene_expression_matrix.csv")

    print("1. Загрузка метаданных...")
    try:
        metadata = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл метаданных не найден по пути {metadata_path}")
        return

    all_samples_dfs = []
    print(f"2. Начало обработки {len(metadata)} файлов...")

    for index, row in metadata.iterrows():
        file_id = row['file_id']
        full_file_path = os.path.join(raw_data_path, file_id)
        
        try:
            # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: ЧИТАЕМ ПРАВИЛЬНЫЙ ФОРМАТ ---
            df = pd.read_csv(
                full_file_path,
                sep='\t',
                header=None,
                skiprows=4,
                # 1. Указываем имена для ВСЕХ 4 колонок
                names=['gene_id_version', 'unstranded', 'stranded_first', 'stranded_second']
            )
            
            # Мы используем 'unstranded' счетчики для нашего анализа
            df_selected = df[['gene_id_version', 'unstranded']].copy()

            # 2. ПРИНУДИТЕЛЬНО конвертируем колонку в string ПЕРЕД использованием .str
            df_selected['gene_id'] = df_selected['gene_id_version'].astype(str).str.split('.').str[0]
            
            # Группируем по очищенному ID и суммируем
            df_aggregated = df_selected.groupby('gene_id')['unstranded'].sum().reset_index()
            
            df_aggregated.set_index('gene_id', inplace=True)
            df_aggregated.rename(columns={'unstranded': file_id}, inplace=True)
            
            all_samples_dfs.append(df_aggregated)
            
        except Exception as e:
            print(f"  - ОШИБКА при обработке файла {file_id}: {e}")
            # Прерываем цикл в случае ошибки, чтобы не выводить 20 одинаковых сообщений
            break 

    if len(all_samples_dfs) != len(metadata):
        print("\nОШИБКА: Обработка файлов была прервана. Финальная матрица не будет создана.")
        return
        
    print("3. Объединение всех образцов в единую матрицу...")
    final_matrix = pd.concat(all_samples_dfs, axis=1)
    
    print("4. Сохранение итоговой матрицы...")
    final_matrix.fillna(0, inplace=True) 
    final_matrix.to_csv(output_path)
    
    print(f"\nУСПЕХ! Матрица экспрессии сохранена в:")
    print(output_path)
    print(f"Размер матрицы: {final_matrix.shape[0]} генов x {final_matrix.shape[1]} образцов")


if __name__ == "__main__":
    process_downloaded_data()