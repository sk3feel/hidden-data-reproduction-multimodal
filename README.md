# Проект DocVQA Privacy Stage 1-2

Проект посвящен первому этапу исследования обезличивания документов в задаче Document VQA.
На текущем этапе реализован базовый экспериментальный контур: подготовка данных, поиск скрываемого значения в OCR и применение обезличивания к тексту и изображению.

## Что реализовано

1. Загрузка и просмотр выборки DocVQA
2. Локализация ответа в OCR токенах
3. Формирование области маскирования на изображении
4. Маскирование найденного фрагмента в изображении
5. Редактирование OCR в диапазоне найденного ответа
6. Оценка доли случаев, где ответ успешно локализован

## Структура проекта

```text
project/
  notebooks/
    01_inspect_docvqa.ipynb
    02_anonymization_demo.ipynb
  src/
    load_data.py
    anonymize.py
  outputs/
    examples/
    stats/
  requirements.txt
  README.md
```

## Запуск

Linux или WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Дальше по порядку:

1. `notebooks/01_inspect_docvqa.ipynb`
2. `notebooks/02_anonymization_demo.ipynb`

## Основные модули и их функции

### `src/load_data.py`

`load_docvqa(split="validation", limit=None)`
- Загружает примеры из DocVQA.
- Поддерживает ограничение количества примеров для быстрых прогонов.

`extract_ocr_tokens(example)`
- Извлекает OCR в плоский список токенов с координатами.
- Используется как единая точка входа OCR для всех дальнейших шагов.

`show_example(example)`
- Выводит вопрос, ответы и часть OCR токенов.
- Показывает изображение для ручной проверки примера.

### `src/anonymize.py`

`normalize_text(text)`
- Приводит текст к единому виду для сопоставления.

`find_answer_span(ocr_tokens, answer)`
- Ищет ответ в OCR последовательности.
- Возвращает границы найденного диапазона токенов и флаг успешного поиска.

`span_bbox_from_tokens(token_entries, start_idx, end_idx)`
- По диапазону токенов строит общий прямоугольник для маскирования.

`mask_image(image, bbox)`
- Применяет маску к найденной области изображения.

`redact_ocr_tokens(ocr_tokens, start_idx, end_idx, strategy="drop")`
- Обезличивает соответствующий диапазон OCR токенов.

`evaluate_match_rate(dataset_subset, output_path=...)`
- Считает долю примеров, в которых ответ удалось локализовать в OCR.
- Сохраняет результат в `outputs/stats/answer_match_rate.json`.

## Что показывают ноутбуки

`01_inspect_docvqa.ipynb`
- Просто загрузка датасета и просмотр структуры.

`02_anonymization_demo.ipynb`
- Подсчет `match_rate`.
- Визуальные примеры до и после маскирования.
- Примеры, где ответ не найден в OCR.

## Текущий статус

Реализация покрывает этап подготовки данных и базового обезличивания.
Модельный этап и итоговая оценка качества DocVQA по метрикам EM и F1 будут выполняться на следующих шагах работы.
