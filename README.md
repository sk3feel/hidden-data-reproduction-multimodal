# Privacy Attacks on Generative Document AI Models

Проект исследует privacy-риски после fine-tuning генеративных мультимодальных моделей на документах из `pixparse/docvqa-single-page-questions`.

Основная идея:
- дообучить generative VLM на `seen` документах;
- проверить `membership inference attack (MIA)` и `data extraction` на `seen` и `unseen`;
- сравнить влияние размера модели, сценария image redaction, OCR-контекста и типа поля;
- собрать воспроизводимый pipeline для локального запуска и Google Colab.

## Исследовательские вопросы

1. Можно ли по поведению модели определить, видела ли она конкретный документ при fine-tuning?
2. Может ли модель восстановить скрытое значение после image/OCR redaction?
3. Какие типы полей наиболее уязвимы к memorization и extraction?
4. Как на privacy-risk влияют размер модели, число эпох, сценарий редактирования и режим `image_only` vs `image+OCR`?

## Модели

| Модель | Размер | Режим | Fine-tuning |
|---|---:|---|---|
| `microsoft/Florence-2-base` | ~230M | `image_only` | full fine-tuning |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | `image_only`, `image+OCR` | QLoRA |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | `image_only` | QLoRA |

Практическая логика эксперимента:
- `Florence-2` используется как компактный image-only baseline;
- `Qwen2-VL-2B` и `Qwen2-VL-7B` сравниваются как более крупные multimodal модели;
- baseline MIA и baseline extraction считаются на базовых pretrained-весах;
- fine-tuned checkpoints используются только для post-training privacy evaluation.

## Данные

Базовый датасет:
- `pixparse/docvqa-single-page-questions`

Ключевые подвыборки:
- `seen/train`: `800` документов для fine-tuning;
- `unseen/validation`: `1612` документов для контроля и privacy evaluation.

Основные артефакты:
- `artifacts/docqa_recovery/benchmark_train/manifest.jsonl`
- `artifacts/docqa_recovery/benchmark/manifest.jsonl`
- `artifacts/finetuning_generative/train_florence2.jsonl`
- `artifacts/finetuning_generative/validation_florence2.jsonl`
- `artifacts/finetuning_generative/train_qwen2vl.jsonl`
- `artifacts/finetuning_generative/validation_qwen2vl.jsonl`

Manifest хранит:
- `example_id`
- `question`
- `answer`
- `answers`
- `field_type`
- `coarse_field_type`
- `answer_spans`
- `answer_bboxes`
- `ocr_tokens`
- `image_path`

Исходные изображения:
- `artifacts/docqa_recovery/benchmark_train/images/original/`
- `artifacts/docqa_recovery/benchmark/images/original/`

## Сценарии анонимизации

### Image-only

Используются:
- `original`
- `img_black`
- `img_white`
- `img_blur_10`
- `img_blur_20`
- `img_blur_50`

### OCR

Поддерживаются базовые OCR-трансформации:
- `ocr_none`
- `ocr_drop_k0`
- `ocr_drop_k20`
- `ocr_mask_k0`
- `ocr_mask_k20`

### Комбинированные image+OCR

В extraction-экспериментах используются сценарии вида:
- `img_black + ocr_mask_k0`
- `img_none + ocr_mask_k0`
- и другие комбинации на уровне кода в `src/inference_scenarios.py`

Важно:
- в `image+OCR` режимах OCR всегда генерируется через `generate_scenario_ocr(...)`;
- для analysis сохраняются `ocr_text_used` и `ocr_source`.

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
- `AUC-ROC` по confidence
- `AUC-ROC` по loss
- `t-test`
- `mean/std` для confidence и loss

Практическая интерпретация:
- baseline MIA должен быть близок к `0.5`;
- baseline extraction должен быть близок к `0`;
- рост MIA/extraction после fine-tuning трактуется как privacy-risk signal.

## Текущее состояние ноутбуков

### `20_florence2_finetune.ipynb`

Назначение:
- full fine-tuning `Florence-2` на `800` train-примерах;
- sanity-check на `seen/unseen`;
- upload checkpoints в HF model repo;
- train loss и sanity в Comet.

Текущая конфигурация:
- epochs: `30`
- checkpoints: `1`, `5`, `15`, `30`

### `21_qwen2vl_finetune.ipynb`

Назначение:
- последовательный QLoRA fine-tuning `Qwen2-VL-2B` и `Qwen2-VL-7B`;
- checkpoint upload в HF;
- финальный sanity-check.

Текущая конфигурация:
- train subset: `200` стратифицированных seen-примеров
- `Qwen2-VL-2B`:
  - epochs: `20`
  - batch size: `1`
  - gradient accumulation: `4`
  - checkpoints: `5`, `10`, `20`
- `Qwen2-VL-7B`:
  - epochs: `10`
  - batch size: `1`
  - gradient accumulation: `4`
  - checkpoints: `5`, `10`

### `22_membership_inference.ipynb`

Назначение:
- baseline MIA на базовых моделях;
- MIA на fine-tuned checkpoints;
- summary tables и figures;
- экспорт результатов в Google Drive и Comet.

Текущая конфигурация:
- `Florence seen`: `200` стратифицированных train-примеров
- `Qwen seen`: те же `200` примеров, что использовались в notebook `21`
- `unseen`: `200` стратифицированных validation-примеров
- fine-tuned checkpoints:
  - `florence2 -> epoch_30`
  - `qwen2b -> epoch_20`
  - `qwen7b -> epoch_10`

Дополнительно:
- в начале ноутбука автоматически восстанавливаются:
  - `qwen2b_sanity_final.csv`
  - `qwen7b_sanity_final.csv`

### `23_data_extraction.ipynb`

Назначение:
- baseline extraction;
- extraction на fine-tuned checkpoints;
- image-only и image+OCR режимы;
- summary tables и heatmaps;
- экспорт результатов в Google Drive и Comet.

Текущая конфигурация:
- `seen/unseen` согласованы с notebook `22`
- fine-tuned checkpoints:
  - `florence2 -> epoch_30`
  - `qwen2b -> epoch_20`
  - `qwen7b -> epoch_10`

### `24_final_analysis.ipynb`

Назначение:
- агрегировать CSV из MIA и extraction;
- строить финальные publication-style figures и tables;
- сохранять итоговые артефакты в Google Drive и Comet.

## Где что хранится

### GitHub

В GitHub хранятся:
- `src/`
- `notebooks/`
- `README.md`
- `requirements.txt`
- лёгкие служебные артефакты

В GitHub не хранятся:
- benchmark images
- большие generated artifacts
- privacy result CSV
- analysis outputs
- checkpoints

### Hugging Face Dataset Repo

Dataset repo:
- `sk3feel/docvqa-privacy-data`

Туда загружается:
- `artifacts_bundle.tar.gz`

Внутри архива:
- `benchmark_train`
- `benchmark`
- `finetuning_generative`

Используется для переноса исходных данных в Colab.

### Hugging Face Model Repo

Model repo:
- `sk3feel/docvqa-privacy-checkpoints`

Там лежат:
- Florence full checkpoints
- Qwen LoRA adapters

Ключевые рабочие checkpoints сейчас:
- `florence2/epoch_30`
- `qwen2b/epoch_20`
- `qwen7b/epoch_10`

### Google Drive

Для notebook `22–24` итоговые артефакты сохраняются в:
- `/content/drive/MyDrive/course_work2026/artifacts/`

Структура:

```text
MyDrive/course_work2026/artifacts/
├── finetuning_generative/
│   ├── qwen_train_200_ids.csv
│   ├── qwen2b_sanity_final.csv
│   └── qwen7b_sanity_final.csv
├── privacy_attacks/
│   ├── mia/
│   └── extraction/
└── analysis/
    ├── figures/
    ├── tables/
    └── qualitative/
```

Это основной канал обмена таблицами и графиками между notebook `22`, `23` и `24`.

### Comet

Comet используется как дополнительный experiment tracker и backup артефактов.

Текущая конфигурация:
- `COMET_WORKSPACE = scfeel`
- `COMET_PROJECT_NAME = qwen3-1`

В Comet логируются:
- train loss
- sanity tables
- MIA CSV и figures
- extraction CSV и figures
- финальные analysis tables и figures

## Как запускать в Colab

Общий порядок:
1. Запустить `20_florence2_finetune.ipynb`
2. Запустить `21_qwen2vl_finetune.ipynb`
3. Запустить `22_membership_inference.ipynb`
4. Запустить `23_data_extraction.ipynb`
5. Запустить `24_final_analysis.ipynb`

Если `20` и `21` уже отработали, для повторного анализа обычно достаточно:
1. `22`
2. `23`
3. `24`

### Важные замечания

- `20` и `21` не используют Google Drive как основной storage layer.
- `22–24` используют Google Drive для CSV, figures и analysis outputs.
- checkpoints остаются в HF model repo и при необходимости скачиваются в `checkpoints/`.
- после restart runtime secrets-ячейку нужно запускать заново.

## Секреты и переменные окружения

Минимально нужны:

```env
HF_TOKEN=""
COMET_API_KEY=""
COMET_WORKSPACE="scfeel"
COMET_PROJECT_NAME="qwen3-1"
COURSE_WORK2026_REPO_URL="https://github.com/sk3feel/hidden-data-reproduction-multimodal.git"
```

Практика:
- в браузерном Colab можно использовать Secrets UI;
- в VS Code + Colab kernel надёжнее задавать `os.environ[...]` в первой ячейке;
- реальные токены нельзя сохранять в `.ipynb`.

## Структура репозитория

```text
course_work2026/
├── notebooks/
│   ├── 20_florence2_finetune.ipynb
│   ├── 21_qwen2vl_finetune.ipynb
│   ├── 22_membership_inference.ipynb
│   ├── 23_data_extraction.ipynb
│   └── 24_final_analysis.ipynb
├── src/
│   ├── anonymize.py
│   ├── colab_setup.py
│   ├── docqa_benchmark.py
│   ├── docqa_metrics.py
│   ├── inference_scenarios.py
│   ├── upload_to_hf.py
│   └── download_from_hf.py
├── artifacts/
│   ├── docqa_recovery/
│   ├── field_labeling/
│   └── finetuning_generative/
├── requirements.txt
└── README.md
```

## Практические выводы по пайплайну

- `20–21` отвечают за обучение и checkpoints.
- `22` даёт результаты по MIA.
- `23` даёт результаты по extraction.
- `24` собирает финальные графики и таблицы для курсовой.
- Для итогового отчёта критичнее всего не train logs, а сохранность CSV и figures из `22–24`.

## Источники форматов моделей

- Florence-2 docs: `https://huggingface.co/docs/transformers/model_doc/florence2`
- Qwen2-VL docs: `https://huggingface.co/docs/transformers/model_doc/qwen2_vl`
- Comet docs: `https://www.comet.com/docs/v2/`
