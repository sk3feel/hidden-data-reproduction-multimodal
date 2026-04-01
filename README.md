# Privacy Attacks on Generative Document AI Models

Репозиторий курсовой работы о privacy-рисках после fine-tuning генеративных мультимодальных моделей на задачах document question answering.

## Идея проекта

- Взять документный QA-датасет
- Привести его к задаче восстановления чувствительных полей документа
- Дообучить несколько генеративных моделей на `seen` документах
- Измерить, возникают ли после этого privacy leakage через:
  - `membership inference attack (MIA)`
  - `data extraction attack`
- Сравнить результаты в двух режимах:
  - `v1` — practical setup
  - `v2` — controlled setup

## Модели

| Model | Size | Input Mode | Fine-tuning |
|---|---:|---|---|
| `microsoft/Florence-2-base` | ~230M | image-only | full fine-tuning |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | image-only, image+OCR | QLoRA |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | image-only | QLoRA |

## Структура проекта

```
src/                          # Основной код
  anonymize.py                # Анонимизация данных
  docqa_benchmark.py          # Построение benchmark
  docqa_metrics.py            # Метрики (EM, F1, corrected)
  inference_scenarios.py      # Сценарии redaction
  load_data.py                # Загрузка данных
  label_docvqa_gigachat.py    # Разметка полей через GigaChat
  prepare_generative_data.py  # Подготовка данных для fine-tuning
  upload_to_hf.py             # Загрузка на Hugging Face
  download_from_hf.py         # Скачивание с Hugging Face
  colab_setup.py              # Настройка Colab окружения

notebooks/                    # Jupyter-ноутбуки экспериментов
  01_inspect_docvqa.ipynb            # Осмотр датасета
  02_anonymization_demo.ipynb        # Демо анонимизации
  03_gigachat_field_labeling.ipynb   # Пилотная разметка полей
  04_gigachat_field_labeling_scalable.ipynb  # Масштабируемая разметка

  # v1 — practical setup
  20_florence2_finetune.ipynb
  21_qwen2vl_finetune.ipynb
  22_membership_inference.ipynb
  23_data_extraction.ipynb
  24_final_analysis.ipynb

  # v2 — controlled setup
  20_florence2_finetune_v2.ipynb
  21_qwen2vl_finetune_v2.ipynb
  22_membership_inference_v2.ipynb
  23_data_extraction_v2.ipynb
  24_final_analysis_v2.ipynb

my_latex/                     # LaTeX-исходники курсовой работы
artifacts/                    # Артефакты разметки и benchmark
```

## Датасет

Базовый датасет: `pixparse/docvqa-single-page-questions` — документный QA по одиночным страницам с OCR, вопросами и ответами.

## Хранение артефактов

- **Git** — код, ноутбуки, небольшие артефакты
- **Hugging Face** — датасет и checkpoints моделей
- **Google Drive** — итоговые артефакты экспериментов (`v1`: `artifacts/`, `v2`: `artifacts_v2_controlled/`)
