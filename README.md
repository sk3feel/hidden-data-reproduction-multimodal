# Privacy Attacks on Generative Document AI Models

Проект исследует privacy-риски после fine-tuning генеративных мультимодальных моделей на документах из `pixparse/docvqa-single-page-questions`.

Главная идея проекта:
- дообучить генеративные VLM на `seen` документах;
- проверить `membership inference attacks (MIA)` и `data extraction` на `seen` и `unseen` данных;
- понять, как на риски влияют размер модели, тип поля, image redaction и наличие OCR-контекста;
- собрать воспроизводимый pipeline, который одинаково запускается локально, в Google Colab и через Colab kernel в VS Code.

Сейчас репозиторий покрывает не только подготовку данных, но и весь экспериментальный цикл:
- сборку benchmark и сценариев анонимизации;
- подготовку generative training JSONL;
- fine-tuning `Florence-2` и `Qwen2-VL`;
- MIA и extraction эксперименты;
- финальный анализ с публикационными графиками;
- хранение данных на `Hugging Face Hub`;
- хранение метрик и аналитических артефактов в `Comet`.

## Исследовательские вопросы

Проект отвечает на такие вопросы:

1. Можно ли по поведению модели определить, видела ли она конкретный документ при fine-tuning?
2. Может ли модель восстановить скрытое значение после image/OCR redaction?
3. Какие типы полей наиболее уязвимы к memorization и extraction?
4. Как зависят privacy-риски от:
- размера модели;
- количества эпох;
- типа поля;
- image-сценария;
- OCR-сценария;
- режима `image_only` vs `image+OCR`?

## Модели и стратегия fine-tuning

| Модель | Размер | Режим | Fine-tuning |
|---|---:|---|---|
| `florence-community/Florence-2-base` | ~230M | `image_only` | full fine-tuning |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | `image_only`, `image+OCR` | QLoRA |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | `image_only`, `image+OCR` | QLoRA |

Текущая логика:
- `Florence-2` используется как компактный image-only generative baseline;
- `Qwen2-VL-2B` и `Qwen2-VL-7B` сравниваются как более мощные мультимодальные модели;
- baseline MIA и baseline extraction всегда считаются на базовых pretrained-весах, без adapter-ов;
- дообученные checkpoints используются только для post-finetuning privacy evaluation.

## Данные

Базовый датасет:
- `pixparse/docvqa-single-page-questions`

Ключевые подвыборки:
- `seen/train`: `800` документов для fine-tuning;
- `unseen/validation`: `1612` документов для контроля и post-train privacy evaluation.

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

### Что хранится в benchmark manifest

В manifest хранятся:
- `example_id`
- `split`
- `question`
- `answer`
- `answers`
- `field_type`
- `coarse_field_type`
- `answer_spans`
- `answer_bboxes`
- `ocr_tokens`
- `image_path`
- дополнительные служебные поля для восстановления сценариев и анализа

Оригинальные изображения:
- `artifacts/docqa_recovery/benchmark/images/original/`
- `artifacts/docqa_recovery/benchmark_train/images/original/`

## Анонимизация и сценарии

### Image-only сценарии

Поддерживаются:
- `original`
- `img_black`
- `img_white`
- `img_blur_10`
- `img_blur_20`
- `img_blur_50`

### OCR-сценарии

Поддерживаются:
- `ocr_none`
- `ocr_drop_k0`
- `ocr_drop_k20`
- `ocr_mask_k0`
- `ocr_mask_k20`

### Комбинированные image+OCR сценарии

Используются, например:
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
- blur с произвольным `sigma` поддерживается на уровне кода;
- для `image+OCR` режимов OCR всегда генерируется через `generate_scenario_ocr(...)`, а не берется как исходный полный OCR;
- в prediction tables сохраняются `ocr_text_used` и `ocr_source`, чтобы было видно, какой текст реально был подан модели.

## Field labeling

Полевая разметка строится гибридно:
- rule-based prelabeling;
- LLM fallback;
- аудит и gold evaluation.

Основные артефакты:
- `artifacts/field_labeling/merged/pixparse__docvqa-single-page-questions__all_splits.csv`
- `artifacts/field_labeling/merged/pixparse__docvqa-single-page-questions__all_splits.jsonl`
- `artifacts/field_labeling/runs/validation/pixparse__docvqa-single-page-questions__validation__field_labels_v1.csv`
- `artifacts/field_labeling/gold_annotation/docvqa_gold_labels_v1.csv`

Ключевые label-поля:
- `field_type`
- `field_group`
- `sensitivity`
- `label_source`
- `rule_reason`
- `rule_confidence`

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
- `Florence-2`: `image_path`, VQA prompt, answer;
- `Qwen2-VL`: `image_path`, `chat_messages`, answer.

## Метрики

### Extraction

Основные:
- `exact_match`
- `token_f1`
- `random_em`
- `random_f1`
- `corrected_em = exact_match - random_em`
- `corrected_f1 = token_f1 - random_f1`

### MIA

Основные:
- `AUC-ROC` по generation confidence
- `AUC-ROC` по loss on gold answer
- `t-test`
- `mean/std` для confidence и loss

Практический смысл:
- baseline MIA до fine-tuning должен быть близок к `0.5`;
- baseline extraction до fine-tuning должен быть близок к `0`;
- рост MIA/extraction после fine-tuning интерпретируется как privacy-risk signal.

## Где что хранится

### GitHub

В GitHub хранятся только легкие и важные для воспроизводимости файлы:
- `src/`
- `notebooks/`
- `README.md`
- `requirements.txt`
- `.env.example`
- benchmark manifests и `summary.json`
- `artifacts/field_labeling/`
- служебные скрипты аудита и sanity-check

В GitHub не хранятся:
- benchmark images;
- `privacy_attacks/` результаты;
- `analysis/` outputs;
- большие JSONL и прочие generated artifacts.

Это сделано намеренно, чтобы репозиторий оставался легким и push не падал на гигабайтах картинок.

### Hugging Face Dataset Repo

Dataset repo:
- `sk3feel/docvqa-privacy-data`

Туда уходит один архив:
- `artifacts_bundle.tar.gz`

Внутри архива лежат:
- `benchmark_train`
- `benchmark`
- `finetuning_generative`

Это основной способ переносить данные в Colab без `Google Drive`.

Сейчас upload/download устроены так:
- `src/upload_to_hf.py` собирает `tar.gz` и грузит его как один файл через `upload_file`;
- `src/download_from_hf.py` скачивает архив через `hf_hub_download`, распаковывает и удаляет локальный `tar.gz`.

### Hugging Face Model Repo

Model repo:
- `sk3feel/docvqa-privacy-checkpoints`

Туда грузятся:
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
- checkpoint naming без zero-padding;
- все notebooks `20–23` уже настроены на `epoch_1`, а не `epoch_01`.

### Comet

Comet используется для experiment tracking и аналитических артефактов.

Текущая конфигурация:
- `COMET_WORKSPACE = scfeel`
- `COMET_PROJECT_NAME = qwen3-1`

В Comet логируются:
- train loss;
- sanity-check таблицы;
- MIA tables и figures;
- extraction tables и figures;
- summary CSV;
- финальные графики и таблицы анализа.

Это важно, потому что `24_final_analysis.ipynb` может подтянуть CSV из Comet fallback-ом, если локальных `artifacts/privacy_attacks/...` уже нет.

## Как работает запуск в новой среде

Проект рассчитан на:
- локальный запуск;
- запуск в Google Colab;
- запуск через Google Colab kernel из VS Code.

### Базовая логика

1. Клонируется репозиторий.
2. `src/colab_setup.py` ставит зависимости.
3. Данные скачиваются из `sk3feel/docvqa-privacy-data`.
4. Делается login в HF.
5. Поднимается Comet experiment в `scfeel/qwen3-1`.

За счет этого:
- не нужен `Google Drive`;
- не нужно вручную копировать benchmark images;
- не нужно заново пересобирать benchmark в Colab.

### Что происходит после restart runtime

Если runtime перезапустился:
- benchmark и generative JSONL снова скачиваются из HF dataset repo;
- checkpoints снова берутся из HF model repo;
- если локальные CSV результатов потеряны, `24_final_analysis.ipynb` пытается забрать их из Comet.

То есть вся рабочая схема опирается на три хранилища:
- GitHub для кода;
- HF dataset/model repos для data и checkpoints;
- Comet для логов и результатов.

## Секреты и переменные окружения

Шаблон:
- `.env.example`

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

Важно:
- `.env` не должен коммититься;
- в браузерном Colab можно использовать Secrets UI;
- в VS Code + Colab kernel `google.colab.userdata.get(...)` может не работать, поэтому там надежнее задавать `os.environ[...]` в первой ячейке;
- после restart runtime env-ячейку нужно выполнить заново.

## Что запускать в Colab

Минимально:
1. Задать env variables или secrets.
2. Выполнить setup-ячейку notebook.
3. Запустить основной pipeline notebook.

Если Colab clone устарел после force-push:

```python
!cd /content/course_work2026 && git fetch origin && git reset --hard origin/main
```

Если нужно начать clean rerun:

```python
!rm -rf /content/course_work2026
```

и потом снова выполнить setup-ячейку.

## Порядок экспериментов

Рекомендуемая последовательность:

1. Подготовить benchmark и generative JSONL локально.
2. Залить dataset bundle на HF:
   `src/upload_to_hf.py`
3. Запустить `20_florence2_finetune.ipynb`
4. Запустить `21_qwen2vl_finetune.ipynb`
5. Запустить `22_membership_inference.ipynb`
6. Запустить `23_data_extraction.ipynb`
7. Запустить `24_final_analysis.ipynb`

## Ноутбуки

### `20_florence2_finetune.ipynb`

Назначение:
- setup через `src/colab_setup.py`;
- загрузка `train_florence2.jsonl` и `validation_florence2.jsonl`;
- full fine-tuning Florence-2;
- upload checkpoints в HF model repo;
- train loss и sanity-check в Comet.

Важные детали:
- используется `florence-community/Florence-2-base`, а не старый remote-code путь;
- prompt формируется как `"<VQA>{question}"`;
- для stability сейчас обучение идет консервативнее:
  - `fp32` train без mixed precision;
  - `batch_size=2`;
  - `LEARNING_RATE=1e-5`;
  - `gradient clipping`;
  - skip non-finite batches;
  - обрезанные PNG читаются через `ImageFile.LOAD_TRUNCATED_IMAGES = True`;
- если предыдущий run дал `NaN`, нельзя продолжать с уже испорченным `model` в памяти: нужен restart runtime и старт notebook с начала;
- плохие локальные checkpoints перед rerun лучше удалить.

### `21_qwen2vl_finetune.ipynb`

Назначение:
- setup через `src/colab_setup.py`;
- QLoRA fine-tuning для `Qwen2-VL-2B` и `Qwen2-VL-7B`;
- upload adapter-ов в HF model repo;
- логирование в Comet.

Важные детали:
- после `2B` run notebook освобождает GPU память перед `7B`;
- training рассчитан на A100;
- checkpoints сохраняются на эпохах `1, 3, 10, 30`.

### `22_membership_inference.ipynb`

Назначение:
- baseline MIA на базовых моделях;
- MIA на finetuned checkpoints;
- AUC, t-test, statistics, histograms;
- epoch curve;
- CSV в `artifacts/privacy_attacks/mia/`;
- логирование CSV и figure-ов в Comet.

Важные детали:
- baseline runs всегда на base weights;
- finetuned runs явно логируют источник весов;
- если prediction пустой или загрязнен спецтокенами, применяется sanitization.

### `23_data_extraction.ipynb`

Назначение:
- baseline extraction;
- extraction на finetuned checkpoints;
- image-only и image+OCR режимы;
- epoch curve;
- CSV в `artifacts/privacy_attacks/extraction/`;
- логирование в Comet.

Важные детали:
- для `image+OCR` OCR берется только через `generate_scenario_ocr(...)`;
- baseline runs используют base pretrained weights;
- prediction tables содержат `ocr_text_used` и `ocr_source`.

### `24_final_analysis.ipynb`

Назначение:
- агрегировать CSV из `artifacts/privacy_attacks/...`;
- fallback на скачивание CSV из Comet;
- строить publication-quality figures;
- сохранять результат в `artifacts/analysis/`;
- логировать analysis outputs обратно в Comet.

Важные детали:
- сохраняет `PNG + PDF`;
- использует `300 DPI`;
- поддерживает локальный режим и режим анализа на другой машине.

## Основные модули

### `src/anonymize.py`

- нормализация текста;
- локализация answer spans;
- OCR redaction;
- image masking;
- blur с произвольным `sigma`.

### `src/docqa_benchmark.py`

- сборка benchmark manifest;
- coarse type mapping;
- генерация сценариев;
- on-the-fly анонимизация через `scenario_payload(...)`.

### `src/docqa_metrics.py`

- `exact_match`
- `token_f1`
- corrected metrics
- random baselines

### `src/inference_scenarios.py`

- `generate_scenario_image(record, scenario_id) -> PIL.Image`
- `generate_scenario_ocr(record, scenario_id) -> str`

### `src/prepare_generative_data.py`

- читает `benchmark_train` и `benchmark`;
- формирует JSONL для `Florence-2` и `Qwen2-VL`.

### `src/run_docqa_experiments.py`

- CLI для benchmark и базовых запусков.

### `src/upload_to_hf.py`

- собирает `artifacts_bundle.tar.gz`;
- загружает один архив в HF dataset repo;
- удаляет локальный архив после upload.

### `src/download_from_hf.py`

- скачивает `artifacts_bundle.tar.gz` из HF;
- распаковывает в `artifacts/`;
- удаляет локальный архив после extraction.

### `src/colab_setup.py`

- читает секреты из Colab или env;
- ставит зависимости;
- логинит в HF;
- скачивает dataset bundle;
- создает Comet experiment.

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

### Загрузка data bundle на Hugging Face

```powershell
.\.venv\Scripts\python.exe src\upload_to_hf.py --repo-id sk3feel/docvqa-privacy-data
```

### Скачивание data bundle с Hugging Face

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

- Если prediction пустой или загрязнен спецтокенами, notebooks `20`, `21` и `23` делают sanitization перед логированием.
- Для privacy-экспериментов лучше держать HF model repo приватным.
- Если в notebook уже возник `NaN` и модель “отравилась”, продолжать тот же runtime не нужно: лучше restart и rerun с начала.
- Если неудачный run уже успел сохранить сломанный checkpoint, успешный rerun потом перезапишет его по тому же пути в HF.
- Comet нужен не только для графиков в UI, но и как резервное хранилище CSV/figures для `24_final_analysis.ipynb`.
- Через VS Code + Colab kernel безопаснее задавать секреты через `os.environ[...]`, а не рассчитывать на `google.colab.userdata.get(...)`.

## Источники форматов моделей

- Florence-2 docs: `https://huggingface.co/docs/transformers/model_doc/florence2`
- Qwen2-VL docs: `https://huggingface.co/docs/transformers/model_doc/qwen2_vl`
- Comet docs: `https://www.comet.com/docs/v2/`
