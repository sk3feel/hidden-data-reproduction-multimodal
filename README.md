# Privacy Attacks on Generative Document AI Models: Membership Inference and Data Extraction after Fine-tuning

Проект исследует privacy-риски после fine-tuning генеративных мультимодальных моделей на документах из `pixparse/docvqa-single-page-questions`.

Центральная идея:
- дообучить `Florence-2-base`, `Qwen2-VL-2B-Instruct`, `Qwen2-VL-7B-Instruct` на `seen` документах;
- проверить `membership inference attacks (MIA)` и `data extraction` на `seen` и `unseen` данных;
- сравнить влияние размера модели, типа поля, степени redaction и наличия OCR-контекста.

Репозиторий уже содержит не только подготовку данных, но и полный reproducible pipeline:
- benchmark и сценарии анонимизации;
- generative training JSONL;
- Colab setup для запуска в новой среде;
- ноутбуки для fine-tuning, MIA, extraction и финального анализа;
- интеграцию с `Hugging Face Hub` и `Comet`.

## Исследовательская рамка

Проект отвечает на такие вопросы:

1. Можно ли по поведению модели определить, видела ли она конкретный документ при fine-tuning?
2. Может ли модель восстановить скрытое значение после image/OCR redaction?
3. Как зависят риски от:
- размера модели `230M -> 2B -> 7B`;
- количества эпох;
- типа поля;
- image-сценария (`black`, `white`, `blur`);
- наличия OCR-контекста (`image_only` vs `image+OCR`)?

## Модели и настройка fine-tuning

| Модель | Параметры | Режим | Способ fine-tuning |
|---|---:|---|---|
| `microsoft/Florence-2-base` | 230M | `image_only` | full fine-tuning |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | `image_only`, `image+OCR` | QLoRA |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | `image_only`, `image+OCR` | QLoRA |

Используемая схема:
- `Florence-2` обучается как image-only VQA baseline;
- `Qwen2-VL` используется и в `image_only`, и в `image+OCR` сценариях;
- baseline MIA и baseline extraction всегда считаются на базовых pretrained-весах, без адаптеров.

## Данные и benchmark

Базовый источник данных: `pixparse/docvqa-single-page-questions`.

В проекте используются две основные выборки:
- `seen/train`: `800` документов для fine-tuning;
- `unseen/validation`: `1612` документов для контроля.

### Train benchmark

Артефакты:
- `artifacts/docqa_recovery/benchmark_train/manifest.jsonl`
- `artifacts/docqa_recovery/benchmark_train/summary.json`

Размер:
- `total_seen = 800`
- `total_kept = 800`

Распределение coarse типов:
- `AMOUNT`: `134`
- `DATE`: `134`
- `ORG`: `133`
- `PERSON`: `133`
- `ID`: `133`
- `CONTACT_ADR`: `133`

### Validation benchmark

Артефакты:
- `artifacts/docqa_recovery/benchmark/manifest.jsonl`
- `artifacts/docqa_recovery/benchmark/summary.json`

Размер:
- `total_seen = 1811`
- `total_kept = 1612`
- `skipped_no_span = 199`

Распределение coarse типов:
- `AMOUNT`: `314`
- `ORG`: `302`
- `PERSON`: `294`
- `ID`: `288`
- `DATE`: `287`
- `CONTACT_ADR`: `127`

### Что хранится в manifest

В benchmark manifest хранятся:
- `question`
- `gold answer / answers`
- `ocr_tokens`
- `answer_spans`
- `answer_bboxes`
- `bbox`
- `coarse_field_type`
- путь к оригинальному изображению

Оригинальные изображения лежат в:
- `artifacts/docqa_recovery/benchmark/images/original/`
- `artifacts/docqa_recovery/benchmark_train/images/original/`

## Сценарии анонимизации

### Image-only

Поддерживаются сценарии:
- `original`
- `img_black`
- `img_white`
- `img_blur_10`
- `img_blur_20`
- `img_blur_50`

### Image + OCR

Для Qwen2-VL дополнительно поддерживаются OCR-режимы:
- `ocr_none`
- `ocr_drop_k0`
- `ocr_drop_k20`
- `ocr_mask_k0`
- `ocr_mask_k20`

Комбинации сценариев используются в extraction-экспериментах, например:
- `img_black + ocr_drop_k0`
- `img_black + ocr_drop_k20`
- `img_black + ocr_mask_k0`
- `img_black + ocr_mask_k20`
- `img_white + ocr_drop_k0`
- `img_white + ocr_mask_k0`
- `img_blur_20 + ocr_drop_k0`
- `img_blur_20 + ocr_mask_k0`
- `img_none + ocr_drop_k0`
- `img_none + ocr_mask_k0`
- `img_black + ocr_none`

Важно:
- image blur поддерживает произвольный `sigma` на уровне кода;
- в notebook `23_data_extraction.ipynb` OCR для режима `image+OCR` берется через `generate_scenario_ocr(...)`, то есть модель получает именно обезличенный OCR, а не исходный текст;
- в prediction-таблицы сохраняются `ocr_text_used` и `ocr_source`, чтобы было видно, какой OCR реально был подан модели.

## Generative training data

Подготовленные JSONL:
- `artifacts/finetuning_generative/train_florence2.jsonl`
- `artifacts/finetuning_generative/validation_florence2.jsonl`
- `artifacts/finetuning_generative/train_qwen2vl.jsonl`
- `artifacts/finetuning_generative/validation_qwen2vl.jsonl`

Размеры:
- `train_florence2.jsonl`: `800`
- `validation_florence2.jsonl`: `1612`
- `train_qwen2vl.jsonl`: `800`
- `validation_qwen2vl.jsonl`: `1612`

Назначение:
- `Florence-2`: `image_path`, VQA prompt, `answer`
- `Qwen2-VL`: `image_path`, `chat_messages`, `answer`

## Метрики

Основные метрики extraction:
- `exact_match`
- `token_f1`
- `random_em`
- `random_f1`
- `corrected_em = exact_match - random_em`
- `corrected_f1 = token_f1 - random_f1`

Основные метрики MIA:
- `AUC-ROC` по confidence
- `AUC-ROC` по loss
- `t-test`
- `mean/std` для `confidence` и `loss`

Практический смысл:
- baseline MIA до fine-tuning должен быть близок к `0.5`;
- baseline extraction до fine-tuning должен быть близок к `0`;
- рост MIA или extraction после fine-tuning трактуется как privacy-risk signal.

## Где что хранится

Это самый важный operational слой проекта.

### Локально в репозитории

Локально хранятся:
- код;
- notebooks;
- benchmark manifests и изображения в `artifacts/docqa_recovery/...`;
- JSONL для fine-tuning в `artifacts/finetuning_generative/...`;
- результаты атак в `artifacts/privacy_attacks/...`;
- финальные фигуры и таблицы в `artifacts/analysis/...`.

### Hugging Face Dataset Repo

Dataset repo:
- `sk3feel/docvqa-privacy-data`

Туда загружается dataset snapshot через `src/upload_to_hf.py`.

Что именно уходит в dataset repo:
- `artifacts/docqa_recovery/benchmark_train/manifest.jsonl`
- `artifacts/docqa_recovery/benchmark_train/images/original/`
- `artifacts/docqa_recovery/benchmark/manifest.jsonl`
- `artifacts/docqa_recovery/benchmark/images/original/`
- `artifacts/finetuning_generative/train_florence2.jsonl`
- `artifacts/finetuning_generative/train_qwen2vl.jsonl`
- `artifacts/finetuning_generative/validation_florence2.jsonl`
- `artifacts/finetuning_generative/validation_qwen2vl.jsonl`

Это основной способ переноса данных в Colab без `Google Drive`.

### Hugging Face Model Repo

Model repo:
- `sk3feel/docvqa-privacy-checkpoints`

Туда загружаются checkpoints и adapters:
- `florence2/epoch_1`
- `florence2/epoch_3`
- `florence2/epoch_10`
- `florence2/epoch_30`
- `florence2/epoch_50`
- `qwen2b/epoch_1`
- `qwen2b/epoch_3`
- `qwen2b/epoch_10`
- `qwen2b/epoch_30`
- `qwen7b/epoch_1`
- `qwen7b/epoch_3`
- `qwen7b/epoch_10`
- `qwen7b/epoch_30`

Важно:
- имена checkpoint-папок без zero-padding;
- notebooks `20`, `21`, `22`, `23` уже настроены именно на формат `epoch_1`, а не `epoch_01`.

### Comet

Comet используется для experiment tracking и хранения производных артефактов:
- train loss curves;
- sanity-check таблиц;
- MIA score tables;
- extraction prediction tables;
- графиков;
- итоговых таблиц анализа.

Workspace:
- `scfeel`

Project:
- `qwen3-1`

В Comet логируются:
- метрики;
- CSV;
- таблицы;
- фигуры;
- summary-артефакты.

## Как не ломаться при смене среды

Проект рассчитан на два режима:
- локальный запуск;
- Colab / VS Code Colab runtime.

### Источник данных по умолчанию

Для Colab и новой среды основной путь такой:
1. Клонируется репозиторий.
2. Через `src/colab_setup.py` ставятся зависимости.
3. Данные скачиваются из `HF dataset repo`.
4. HF login используется для дальнейшей загрузки чекпоинтов.
5. Comet инициализируется для логирования эксперимента.

Это позволяет не зависеть от `Google Drive`.

### Что делать после перезапуска сессии

Если runtime перезапустился:
- benchmark и generative JSONL заново подтягиваются из `sk3feel/docvqa-privacy-data`;
- checkpoints заново подтягиваются из `sk3feel/docvqa-privacy-checkpoints`;
- если локальные CSV результатов уже отсутствуют, `24_final_analysis.ipynb` сначала пытается читать `artifacts/privacy_attacks/...`, а при отсутствии файлов использует fallback через Comet API.

То есть анализ не привязан к одной конкретной машине, пока:
- HF dataset repo содержит данные;
- HF model repo содержит checkpoints;
- Comet содержит залогированные CSV и фигуры.

### Как работает fallback в финальном анализе

`notebooks/24_final_analysis.ipynb` устроен так:
- основной путь: читать локальные CSV из `artifacts/privacy_attacks/mia/` и `artifacts/privacy_attacks/extraction/`;
- fallback: если CSV отсутствуют, notebook пытается скачать `.csv` assets из Comet по именам experiments внутри `qwen3-1`.

Это важно для случаев, когда:
- analysis запускается на другой машине;
- Colab session была очищена;
- локальные `artifacts/privacy_attacks/...` не сохранились.

## Секреты и переменные окружения

Шаблон находится в `.env.example`.

Ключевые переменные:

```dotenv
HF_TOKEN=""
HF_DATASET_REPO="sk3feel/docvqa-privacy-data"
HF_MODEL_REPO="sk3feel/docvqa-privacy-checkpoints"

COMET_API_KEY=""
COMET_WORKSPACE="scfeel"
COMET_PROJECT_NAME="qwen3-1"

COURSE_WORK2026_REPO_URL="https://github.com/sk3feel/hidden-data-reproduction-multimodal.git"
```

В Colab `src/colab_setup.py` сначала пытается читать секреты через `google.colab.userdata.get(...)`, а затем берет их из окружения.

Это значит:
- локально можно использовать `.env`;
- в Colab можно использовать Secrets UI;
- код работает одинаково в обеих средах.

## Порядок запуска

Рекомендуемая последовательность:

1. Подготовить benchmark и JSONL локально.
2. Залить dataset snapshot на HF:
   `src/upload_to_hf.py`
3. Запустить fine-tuning `Florence-2`:
   `notebooks/20_florence2_finetune.ipynb`
4. Запустить fine-tuning `Qwen2-VL`:
   `notebooks/21_qwen2vl_finetune.ipynb`
5. Прогнать MIA:
   `notebooks/22_membership_inference.ipynb`
6. Прогнать extraction:
   `notebooks/23_data_extraction.ipynb`
7. Собрать графики и таблицы:
   `notebooks/24_final_analysis.ipynb`

## Ноутбуки

### `20_florence2_finetune.ipynb`

Назначение:
- setup через `src/colab_setup.py`;
- загрузка `train_florence2.jsonl`;
- full fine-tuning `Florence-2`;
- сохранение checkpoint-ов в HF model repo;
- логирование train loss и sanity-check в Comet.

Ключевой prompt для `Florence-2`:
- используется VQA task token `"<VQA>"`;
- prompt формируется как `"<VQA>{question}"`.

### `21_qwen2vl_finetune.ipynb`

Назначение:
- setup через `src/colab_setup.py`;
- QLoRA fine-tuning для `Qwen2-VL-2B` и `Qwen2-VL-7B`;
- upload adapters в HF model repo;
- логирование в Comet.

Важно:
- после завершения `2B` run notebook освобождает GPU память перед `7B` через `del ...`, `gc.collect()`, `torch.cuda.empty_cache()`.

### `22_membership_inference.ipynb`

Назначение:
- baseline MIA на базовых pretrained-моделях;
- MIA на finetuned checkpoints;
- `AUC-ROC`, `t-test`, `mean/std`, histogram plots;
- epoch curve по checkpoint-ам;
- сохранение CSV в `artifacts/privacy_attacks/mia/`;
- логирование CSV и фигур в Comet.

Важно:
- baseline runs грузят именно base weights, без adapters;
- finetuned runs явно логируют `weights_source` и `adapter_path`.

### `23_data_extraction.ipynb`

Назначение:
- baseline extraction на базовых моделях;
- extraction на finetuned checkpoints;
- image-only и image+OCR сценарии;
- epoch curve;
- сохранение CSV в `artifacts/privacy_attacks/extraction/`;
- логирование CSV и фигур в Comet.

Важно:
- OCR для `image+OCR` берется через `generate_scenario_ocr(...)`;
- prediction-таблицы содержат `ocr_text_used` и `ocr_source`;
- baseline runs используют base pretrained weights.

### `24_final_analysis.ipynb`

Назначение:
- собрать все CSV из `artifacts/privacy_attacks/...`;
- при необходимости скачать CSV из Comet fallback-ом;
- построить publication-quality графики;
- сохранить фигуры в `PNG` и `PDF` с `300 DPI`;
- сохранить таблицы в `CSV` и `LaTeX`;
- сложить результат в `artifacts/analysis/` и залогировать в Comet.

## Основные модули

### `src/anonymize.py`

- нормализация текста;
- поиск `answer_spans`;
- OCR redaction;
- image masking;
- blur с произвольным `sigma`.

### `src/docqa_benchmark.py`

- сборка benchmark manifest;
- coarse type mapping;
- генерация сценариев;
- on-the-fly анонимизация через `scenario_payload(...)`.

### `src/prepare_generative_data.py`

- читает `benchmark_train` и `benchmark`;
- готовит JSONL для `Florence-2` и `Qwen2-VL`;
- сохраняет данные в `artifacts/finetuning_generative/`.

### `src/inference_scenarios.py`

- `generate_scenario_image(record, scenario_id) -> PIL.Image`
- `generate_scenario_ocr(record, scenario_id) -> str`
- используется в extraction notebooks.

### `src/upload_to_hf.py`

- собирает dataset snapshot из `artifacts/`;
- заливает его в `HF dataset repo`.

### `src/download_from_hf.py`

- скачивает dataset snapshot из `HF dataset repo` обратно в `artifacts/`.

### `src/colab_setup.py`

- клонирует репозиторий, если его нет в runtime;
- ставит зависимости из `requirements.txt`;
- скачивает data snapshot с HF;
- логинит в HF Hub;
- поднимает Comet experiment.

## Структура репозитория

```text
project/
  notebooks/
    01_inspect_docvqa.ipynb
    02_anonymization_demo.ipynb
    03_gigachat_field_labeling.ipynb
    04_gigachat_field_labeling_scalable.ipynb
    20_florence2_finetune.ipynb
    21_qwen2vl_finetune.ipynb
    22_membership_inference.ipynb
    23_data_extraction.ipynb
    24_final_analysis.ipynb
  src/
    anonymize.py
    colab_setup.py
    docqa_benchmark.py
    docqa_metrics.py
    download_from_hf.py
    inference_scenarios.py
    label_docvqa_gigachat.py
    load_data.py
    prepare_generative_data.py
    run_docqa_experiments.py
    upload_to_hf.py
  artifacts/
    field_labeling/
    docqa_recovery/
      benchmark/
      benchmark_train/
      benchmark_smoke/
      scenarios/
      validation/
    finetuning_generative/
    privacy_attacks/
      mia/
      extraction/
    analysis/
```

## Установка локально

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name course_work2026_venv --display-name "Python (course_work2026)"
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m ipykernel install --user --name course_work2026_venv --display-name "Python (course_work2026)"
```

## Полезные команды

### Подготовка generative JSONL

```powershell
.\.venv\Scripts\python.exe src\prepare_generative_data.py
```

### Загрузка dataset snapshot на Hugging Face Hub

```powershell
.\.venv\Scripts\python.exe src\upload_to_hf.py --repo-id sk3feel/docvqa-privacy-data
```

### Скачивание dataset snapshot с Hugging Face Hub

```powershell
.\.venv\Scripts\python.exe src\download_from_hf.py --repo-id sk3feel/docvqa-privacy-data
```

### Аудит анонимизации

```powershell
.\.venv\Scripts\python.exe audit_benchmark_anonymization.py
```

### Sanity-check benchmark

```powershell
.\.venv\Scripts\python.exe sanity_check.py
```

## Практические замечания

- Если модель возвращает пустую строку или остатки спецтокенов, notebooks `20`, `21` и `23` делают sanitization prediction before logging.
- Для privacy-экспериментов checkpoints лучше хранить в приватном HF model repo.
- Для reproducibility полезно хранить в Comet не только метрики, но и CSV/figures, потому что именно они потом нужны `24_final_analysis.ipynb`.
- Если analysis запускается на другой машине, сначала проверь наличие локальных CSV в `artifacts/privacy_attacks/...`; если их нет, notebook `24` должен добрать результаты из Comet.

## Источники форматов моделей

- Florence-2 model card: `https://huggingface.co/microsoft/Florence-2-base`
- Qwen2-VL docs in Transformers: `https://huggingface.co/docs/transformers/model_doc/qwen2_vl`
- Comet Python API docs: `https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/API/`
