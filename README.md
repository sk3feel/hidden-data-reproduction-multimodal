# Privacy Attacks on Generative Document AI Models

Этот репозиторий содержит полный пайплайн курсовой работы по исследованию privacy-рисков после fine-tuning генеративных мультимодальных моделей на задачах документного Question Answering.

Главная идея проекта:
- взять документный QA-датасет;
- привести его к задаче восстановления чувствительных полей документа;
- дообучить несколько генеративных моделей на `seen` документах;
- измерить, возникают ли после этого privacy leakage через:
  - `membership inference attack (MIA)`;
  - `data extraction attack`;
- сравнить результаты в двух режимах:
  - `v1`: practical setup, где каждая модель дообучалась в удобном для неё режиме;
  - `v2`: controlled setup, где условия обучения были выровнены.

README специально сделан максимально подробным. По нему должно быть возможно:
- понять экспериментальный дизайн проекта;
- восстановить структуру артефактов;
- разобраться в разметке полей и coarse field types;
- понять, как строятся train/seen/unseen выборки;
- понять, что именно считается в метриках;
- написать по проекту методологию, раздел экспериментов и обсуждение результатов.

## 1. Research Questions

Проект отвечает на четыре исследовательских вопроса.

1. Можно ли по поведению модели определить, видела ли она конкретный документ во время fine-tuning?
2. Может ли модель восстановить скрытое значение поля после редактирования изображения и/или OCR?
3. Какие типы полей наиболее уязвимы к memorization и extraction?
4. Как на privacy leakage влияют:
   - размер модели;
   - режим fine-tuning;
   - объем обучающей выборки;
   - сценарий redaction?

## 2. High-Level Pipeline

Полный пайплайн устроен так:

1. Загружается исходный датасет `pixparse/docvqa-single-page-questions`.
2. Для каждой пары `question-answer` предсказывается тип поля.
3. Исходные типы сворачиваются в небольшое число coarse categories.
4. Строится benchmark:
   - сохраняются изображения;
   - извлекаются OCR-токены;
   - находятся answer spans;
   - строятся bbox для target field;
   - формируется manifest.
5. Из benchmark строятся generative training sets для Florence и Qwen.
6. Выполняется fine-tuning моделей.
7. Запускается `membership inference`.
8. Запускается `data extraction`.
9. В отдельном ноутбуке строятся финальные таблицы, графики и qualitative examples.

## 3. Dataset

### 3.1 Source Dataset

Базовый датасет:
- `pixparse/docvqa-single-page-questions`

Это датасет документного question answering по одиночным страницам документов. Для каждого примера доступны:
- изображение страницы;
- OCR;
- вопрос;
- один или несколько допустимых ответов.

Почему выбран именно этот датасет:
- он ближе к реальным задачам document AI, чем чисто текстовые приватностные бенчмарки;
- в нем есть поля разной природы:
  - даты;
  - суммы;
  - идентификаторы;
  - имена;
  - организации;
  - адреса/контакты;
- он подходит для построения both pixel-level and OCR-level redaction experiments.

### 3.2 Local Data Sources

В проекте используются как локально подготовленные артефакты, так и данные, сохраненные в `artifacts/`.

Ключевые пути:
- `artifacts/field_labeling/merged/pixparse__docvqa-single-page-questions__all_splits.csv`
- `artifacts/docqa_recovery/benchmark_train/manifest.jsonl`
- `artifacts/docqa_recovery/benchmark/manifest.jsonl`
- `artifacts/finetuning_generative/train_florence2.jsonl`
- `artifacts/finetuning_generative/validation_florence2.jsonl`
- `artifacts/finetuning_generative/train_qwen2vl.jsonl`
- `artifacts/finetuning_generative/validation_qwen2vl.jsonl`

## 4. Field Labeling and Type System

### 4.1 Initial Labeling

Первичная семантическая разметка выполняется через GigaChat в:
- `src/label_docvqa_gigachat.py`
- `notebooks/03_gigachat_field_labeling.ipynb`
- `notebooks/04_gigachat_field_labeling_scalable.ipynb`

Базовый закрытый список типов:
- `DATE`
- `AMOUNT`
- `NAME`
- `ID`
- `ADDRESS`
- `PHONE`
- `ORG`
- `PERCENTAGE`
- `REFERENCE`
- `OTHER`

Если в закрытый список поле не укладывается, модель могла предложить новый тип в `snake_case`.

### 4.2 Coarse Field Types

Для экспериментов используется укрупненная схема типов, заданная в `src/docqa_benchmark.py`:

- `DATE_TIME -> DATE`
- `MONEY -> AMOUNT`
- `QUANTITY -> AMOUNT`
- `PERCENTAGE -> AMOUNT`
- `IDENTIFIER -> ID`
- `DOCUMENT_REFERENCE -> ID`
- `PERSON_NAME -> PERSON`
- `ORG_NAME -> ORG`
- `ADDRESS -> CONTACT_ADR`
- `CONTACT -> CONTACT_ADR`

Финальные coarse field types, на которых строятся benchmark и stratified splits:
- `DATE`
- `AMOUNT`
- `ID`
- `PERSON`
- `ORG`
- `CONTACT_ADR`

### 4.3 Additional Metadata

В labels CSV и manifest дополнительно сохраняются:
- `field_type`
- `coarse_field_type`
- `field_group`
- `sensitivity`

Они позволяют:
- строить стратифицированные выборки;
- сравнивать leakage по типам полей;
- потенциально использовать sensitivity-aware analysis.

## 5. Benchmark Construction

Основная логика построения benchmark находится в:
- `src/docqa_benchmark.py`

### 5.1 Что делает benchmark builder

Для каждого примера:
- находит соответствующий исходный объект датасета по `split` и `local_row_id`;
- извлекает изображение;
- извлекает OCR-токены;
- находит все answer spans в OCR;
- строит bbox для целевого ответа;
- сохраняет изображение локально;
- записывает manifest row.

### 5.2 Когда пример отбрасывается

Пример не попадает в benchmark, если:
- не удалось найти span ответа в OCR;
- отсутствует изображение;
- не удалось получить bbox для ответа.

### 5.3 Что хранится в manifest

Каждая строка manifest содержит:
- `dataset_name`
- `split`
- `local_row_id`
- `example_id`
- `question`
- `answer`
- `answers`
- `field_type`
- `coarse_field_type`
- `field_group`
- `sensitivity`
- `answer_start_idx`
- `answer_end_idx`
- `answer_bbox`
- `answer_spans`
- `answer_bboxes`
- `ocr_tokens`
- `image_path`
- `image_size`

Это критично для всей дальнейшей работы:
- `MIA` использует `question`, `answer`, `coarse_field_type`, `image_path`;
- `Extraction` использует `ocr_tokens`, `answer_spans`, `image_path`, `coarse_field_type`;
- qualitative analysis в финальных ноутбуках строится по этим же данным.

## 6. Redaction Scenarios

Сценарии генерации измененных изображений и OCR описаны в:
- `src/inference_scenarios.py`

### 6.1 Image Scenarios

Поддерживаются:
- `original`
- `img_none`
- `img_black`
- `img_white`
- `img_blur_10`
- `img_blur_20`
- `img_blur_50`

Смысл:
- `img_black`: черная маска поверх целевого поля;
- `img_white`: белая маска;
- `img_blur_*`: размытие целевого поля с разной интенсивностью.

### 6.2 OCR Scenarios

Поддерживаются:
- `original`
- `ocr_none`
- `ocr_drop_k0`
- `ocr_drop_k20`
- `ocr_mask_k0`
- `ocr_mask_k20`

Где:
- `drop` удаляет соответствующие OCR токены;
- `mask` заменяет их на `[REDACTED]`;
- `k_0` и `k_20` задают размер контекстного окна вокруг target span.

### 6.3 Combined Scenarios

В extraction-ноутбуках используются комбинированные сценарии вида:
- `img_black__ocr_mask__k_0`
- `img_none__ocr_mask__k_0`

Смысл этих сценариев:
- можно отдельно проверить visual memorization;
- можно проверить, помогает ли модели OCR;
- можно понять, утечка идет по изображению, по OCR или по обоим каналам.

## 7. Generative Formulation

### 7.1 Florence

`Florence-2-base` обучается как image-only generative VQA модель.

Формат задачи:
- вход: изображение + текст `"<VQA>{question}"`;
- выход: строковый ответ.

### 7.2 Qwen2-VL

`Qwen2-VL-2B` и `Qwen2-VL-7B` обучаются как chat-style multimodal модели.

Формат user message:
- image;
- text-instruction;
- вопрос;
- иногда OCR text.

Варианты:
- `image_only`
- `image+OCR`

Для Qwen используется prompt вида:
- ответить только значением;
- без дополнительных пояснений;
- в OCR-ветке OCR добавляется в prompt.

## 8. Models and Fine-Tuning Regimes

В проекте сравниваются три модели.

| Model | Size | Input Mode | Fine-tuning regime |
|---|---:|---|---|
| `microsoft/Florence-2-base` | ~230M | image-only | full fine-tuning |
| `Qwen/Qwen2-VL-2B-Instruct` | 2B | image-only, image+OCR | QLoRA |
| `Qwen/Qwen2-VL-7B-Instruct` | 7B | image-only | QLoRA |

Почему режимы отличаются:
- Florence достаточно маленькая и может быть дообучена full fine-tuning;
- Qwen-модели существенно больше, поэтому для них использовался parameter-efficient fine-tuning через QLoRA;
- это отражает realistic training setup, но создает методологическое ограничение при сравнении моделей.

## 9. Two Experimental Tracks

Проект содержит две экспериментальные ветки.

### 9.1 `v1`: Practical Setup

Это основная практическая ветка, где каждая модель дообучалась в удобной для неё конфигурации.

#### Fine-tuning budget

- `Florence-2`
  - `800` train examples
  - full fine-tuning
  - `30` эпох
  - финальный checkpoint: `epoch_30`

- `Qwen2-VL-2B`
  - `200` stratified train examples
  - QLoRA
  - `20` эпох
  - финальный checkpoint: `epoch_20`

- `Qwen2-VL-7B`
  - те же `200` stratified train examples
  - QLoRA
  - `10` эпох
  - финальный checkpoint: `epoch_10`

#### Что означает `v1`

`v1` отвечает на вопрос:
- что произойдет в practically feasible training setup, если дообучать каждую модель до осмысленной сходимости?

### 9.2 `v2`: Controlled Setup

Это дополнительная ветка для более чистого сравнения.

#### Controlled design

- для всех моделей берется один и тот же `stratified 200` train subset;
- для всех моделей используется один и тот же `unseen 200` validation subset;
- всем моделям дается одинаковый training budget: `10` эпох;
- checkpoints сохраняются на `3`, `5`, `10` эпохах.

#### Fine-tuning budget

- `Florence-2`
  - `200` stratified train examples
  - full fine-tuning
  - `10` эпох

- `Qwen2-VL-2B`
  - те же `200` train examples
  - QLoRA
  - `10` эпох

- `Qwen2-VL-7B`
  - те же `200` train examples
  - QLoRA
  - `10` эпох

#### Баланс coarse types в `v2`

В `v2` стратифицированная выборка распределяется максимально ровно.

Для `200` примеров и `6` coarse types идеальное равенство невозможно, поэтому разница между типами составляет максимум `1` пример.

Типичный паттерн:
- `34`
- `34`
- `33`
- `33`
- `33`
- `33`

#### Что означает `v2`

`v2` отвечает на вопрос:
- сохраняется ли privacy leakage, если сравнивать модели в более честной, выровненной конфигурации?

## 10. Seen / Unseen Splits

### 10.1 Для fine-tuning

Используются:
- train split benchmark;
- validation split benchmark.

### 10.2 Для MIA и Extraction

Используются две группы:
- `seen`
- `unseen`

`seen`:
- примеры, которые использовались в обучении модели;
- именно на них ожидается memorization.

`unseen`:
- стратифицированная подвыборка validation benchmark;
- модель их не видела при fine-tuning.

В `v1`:
- для Florence `seen` — `200` стратифицированных train examples;
- для Qwen `seen` — те же `200`, на которых модель обучалась.

В `v2`:
- для всех моделей `seen` — один и тот же общий `stratified 200`.

## 11. Metrics

### 11.1 Sanity Metrics

Во время fine-tuning проверяется:
- `seen_em`
- `unseen_em`
- промежуточные sanity EM на selected epochs

Зачем:
- убедиться, что модель действительно учится;
- увидеть gap между memorization и generalization;
- отсеять технически сломанные запуски.

### 11.2 Membership Inference Metrics

Основная функция вычисляет:
- `auc_confidence`
- `auc_loss`
- `t_stat`
- `p_value`
- `mean_seen_conf`
- `std_seen_conf`
- `mean_unseen_conf`
- `std_unseen_conf`
- `mean_seen_loss`
- `mean_unseen_loss`

#### Confidence

Интуиция:
- если модель видела пример на обучении, она может быть более уверенной в генерации ответа.

Практическая реализация:
- усреднение лог-вероятностей сгенерированных токенов.

#### Loss

Интуиция:
- на seen-примерах loss должен быть ниже, чем на unseen.

При расчете `AUC` используется `-loss`, потому что:
- lower loss = stronger membership signal.

#### Интерпретация MIA

- `AUC ≈ 0.5`: модель почти не различает seen и unseen;
- `AUC >> 0.5`: membership leakage присутствует;
- чем ближе к `1.0`, тем сильнее leakage.

### 11.3 Extraction Metrics

Для extraction используются:
- `exact_match`
- `token_f1`
- `random_em`
- `random_f1`
- `corrected_em`
- `corrected_f1`

#### Random baseline

Он считается в `src/docqa_metrics.py`.

Идея:
- случайный ответ берется не из всех возможных ответов вообще, а из пула ответов того же `coarse_field_type`;
- это делает baseline более честным.

#### Corrected metrics

Используются:
- `corrected_em = exact_match - random_em`
- `corrected_f1 = token_f1 - random_f1`

Это важно, потому что:
- некоторые типы полей легче угадываются случайно;
- `corrected_*` лучше отражают именно memorization/extraction, а не просто простоту задачи.

#### Интерпретация extraction

- baseline должен быть около `0`;
- высокий `corrected_em` на `seen` означает, что модель извлекает скрытые поля лучше случайного baseline;
- сильный разрыв `seen >> unseen` интерпретируется как сигнал memorization.

## 12. Notebook Map

### 12.1 Preparatory notebooks

- `01_inspect_docvqa.ipynb`
  - первичный осмотр датасета;
- `02_anonymization_demo.ipynb`
  - демонстрация redaction/anonymization;
- `03_gigachat_field_labeling.ipynb`
  - базовая разметка полей;
- `04_gigachat_field_labeling_scalable.ipynb`
  - масштабируемая разметка.

### 12.2 Main experiment notebooks (`v1`)

- `20_florence2_finetune.ipynb`
- `21_qwen2vl_finetune.ipynb`
- `22_membership_inference.ipynb`
- `23_data_extraction.ipynb`
- `24_final_analysis.ipynb`

### 12.3 Controlled experiment notebooks (`v2`)

- `20_florence2_finetune_v2.ipynb`
- `21_qwen2vl_finetune_v2.ipynb`
- `22_membership_inference_v2.ipynb`
- `23_data_extraction_v2.ipynb`
- `24_final_analysis_v2.ipynb`

## 13. Results: `v1`

## 13.1 Fine-Tuning Results

### Florence-2 (`20_florence2_finetune.ipynb`)

- train loss: примерно `1.79 -> 0.019`
- final sanity:
  - `seen_em = 0.96`
  - `unseen_em = 0.10`

Интерпретация:
- Florence в `v1` очень сильно запомнила train set;
- generalization на unseen заметно ниже.

### Qwen2-VL-2B (`21_qwen2vl_finetune.ipynb`)

- train loss: `1.5801 -> 0.0286`
- sanity EM:
  - `EM@5 = 0.30`
  - `EM@10 = 0.35`
  - `EM@20 = 0.80`
- final:
  - `seen_em = 0.92`
  - `unseen_em = 0.16`

### Qwen2-VL-7B (`21_qwen2vl_finetune.ipynb`)

- train loss: `1.4652 -> 0.1118`
- sanity EM:
  - `EM@5 = 0.30`
  - `EM@10 = 0.65`
- final:
  - `seen_em = 0.80`
  - `unseen_em = 0.21`

### Fine-tuning summary for `v1`

- все три модели сошлись;
- все три модели запоминают train лучше, чем обобщают на validation;
- strongest memorization по sanity в `v1` показывают Florence и Qwen-2B.

## 13.2 MIA Results (`22_membership_inference.ipynb`)

### Baseline

| Model | AUC confidence | AUC loss |
|---|---:|---:|
| Florence-2 baseline | `0.572125` | `0.513950` |
| Qwen2-VL-2B baseline | `0.500000` | `0.549225` |

### Fine-tuned

| Model | Checkpoint | AUC confidence | AUC loss |
|---|---|---:|---:|
| Florence-2 | `epoch_30` | `0.577788` | `0.992000` |
| Qwen2-VL-2B | `epoch_20` | `0.500000` | `0.963775` |
| Qwen2-VL-7B | `epoch_10` | `0.500000` | `0.939875` |

### Interpreтация `v1` MIA

- baseline почти не показывает сильного membership leakage;
- после fine-tuning `loss-based MIA` становится очень сильным у всех моделей;
- strongest `loss-based MIA` показывает Florence;
- `confidence-based MIA` для Qwen в `v1` невалиден:
  - `confidence` схлопнулся в константу;
  - `auc_confidence = 0.5`;
  - `p_value = NaN`.

Поэтому в `v1` для Qwen корректно интерпретировать только:
- `auc_loss`
- field-type MIA по `loss`

## 13.3 Extraction Results (`23_data_extraction.ipynb`)

### Florence baseline

Baseline Florence почти не извлекает скрытые ответы:
- corrected metrics близки к `0`;
- это важный sanity baseline.

### Florence fine-tuned (`epoch_30`)

Image-only, `seen corrected_em`:
- `original = 0.96575`
- `img_black = 0.14575`
- `img_blur_20 = 0.15075`
- `img_blur_50 = 0.15575`

### Qwen2-VL-2B fine-tuned (`epoch_20`, image-only)

- `original = 0.94575`
- `img_black = 0.59575`
- `img_blur_20 = 0.60075`
- `img_blur_50 = 0.60075`

### Qwen2-VL-2B fine-tuned (`epoch_20`, image+OCR)

- `img_black__ocr_mask__k_0 = 0.02575`
- `img_none__ocr_mask__k_0 = 0.32075`

### Qwen2-VL-7B fine-tuned (`epoch_10`, image-only)

- `original = 0.88075`
- `img_black = 0.54575`
- `img_blur_20 = 0.54575`
- `img_blur_50 = 0.55075`

### Interpretation `v1` Extraction

- baseline Florence почти не течет;
- fine-tuned модели уверенно извлекают скрытые поля на `seen`;
- на masked/blurred images strongest extraction в `v1` показывает `Qwen-2B`;
- OCR-ветка у `Qwen-2B` существенно слабее image-only;
- это указывает, что основной канал leakage в `v1` у Qwen — визуальный.

## 14. Results: `v2`

## 14.1 Fine-Tuning Results

### Florence-2 (`20_florence2_finetune_v2.ipynb`)

Train loss по эпохам:
- `Epoch 1 = 2.0561`
- `Epoch 2 = 1.0603`
- `Epoch 3 = 0.7286`
- `Epoch 4 = 0.5176`
- `Epoch 5 = 0.3787`
- `Epoch 6 = 0.3194`
- `Epoch 7 = 0.2118`
- `Epoch 8 = 0.1385`
- `Epoch 9 = 0.0997`
- `Epoch 10 = 0.0934`

Final sanity:
- `seen_em = 0.90`
- `unseen_em = 0.08`

### Qwen2-VL-2B (`21_qwen2vl_finetune_v2.ipynb`)

Train loss:
- `Epoch 1 = 1.5795`
- `Epoch 2 = 1.3891`
- `Epoch 3 = 1.2312`
- `Epoch 4 = 1.0659`
- `Epoch 5 = 0.8821`
- `Epoch 6 = 0.7231`
- `Epoch 7 = 0.6023`
- `Epoch 8 = 0.5259`
- `Epoch 9 = 0.3915`
- `Epoch 10 = 0.2795`

Sanity:
- `EM@3 = 0.20`
- `EM@5 = 0.30`
- `EM@10 = 0.55`

Final:
- `seen_em = 0.77`
- `unseen_em = 0.20`

### Qwen2-VL-7B (`21_qwen2vl_finetune_v2.ipynb`)

Train loss:
- `Epoch 1 = 1.4733`
- `Epoch 2 = 1.2149`
- `Epoch 3 = 0.8876`
- `Epoch 4 = 0.6467`
- `Epoch 5 = 0.4857`
- `Epoch 6 = 0.3143`
- `Epoch 7 = 0.2083`
- `Epoch 8 = 0.1515`
- `Epoch 9 = 0.1150`
- `Epoch 10 = 0.1182`

Sanity:
- `EM@3 = 0.25`
- `EM@5 = 0.35`
- `EM@10 = 0.60`

Final:
- `seen_em = 0.84`
- `unseen_em = 0.24`

### Fine-tuning summary for `v2`

- все модели снова сошлись;
- `Qwen-7B` в controlled setup выглядит сильнее `Qwen-2B` по final sanity;
- Florence по-прежнему показывает очень сильный memorization signal.

## 14.2 MIA Results (`22_membership_inference_v2.ipynb`)

### Baseline

| Model | AUC confidence | AUC loss |
|---|---:|---:|
| Florence-2 baseline | `0.578688` | `0.516975` |
| Qwen2-VL-2B baseline | `0.500000` | `0.549300` |

### Fine-tuned

| Model | Checkpoint | AUC confidence | AUC loss |
|---|---|---:|---:|
| Florence-2 | `epoch_10` | `0.582750` | `0.988675` |
| Qwen2-VL-2B | `epoch_10` | `0.500000` | `0.920700` |
| Qwen2-VL-7B | `epoch_10` | `0.500000` | `0.941725` |

### Field-type MIA in `v2`

По coarse field types `auc_loss` остается высоким.

Пример для `Qwen-7B`:
- `AMOUNT = 0.855536`
- `CONTACT_ADR = 0.972318`
- `DATE = 0.953168`
- `ID = 0.933884`
- `ORG = 0.962351`
- `PERSON = 0.967860`

### Interpretation `v2` MIA

- controlled setup подтверждает, что loss-based membership leakage не исчезает;
- strongest `MIA loss` снова показывает Florence;
- у обеих Qwen `auc_loss` остается очень высоким;
- `confidence-based MIA` для Qwen в сохраненных outputs `v2` все еще невалиден и не должен использоваться как основной вывод.

## 14.3 Extraction Results (`23_data_extraction_v2.ipynb`)

### Florence baseline

Baseline снова около нуля:
- существенной extraction leakage без fine-tuning нет.

### Florence fine-tuned (`epoch_10`, image-only)

`seen corrected_em`:
- `original = 0.93075`
- `img_black = 0.08075`
- `img_blur_20 = 0.08075`
- `img_blur_50 = 0.08075`

### Qwen2-VL-2B (`epoch_10`, image-only)

`seen corrected_em`:
- `original = 0.80575`
- `img_black = 0.47575`
- `img_blur_20 = 0.48075`
- `img_blur_50 = 0.49075`

### Qwen2-VL-2B (`epoch_10`, image+OCR)

`seen corrected_em`:
- `img_black__ocr_mask__k_0 = 0.01575`
- `img_none__ocr_mask__k_0 = 0.31575`

### Qwen2-VL-7B (`epoch_10`, image-only)

`seen corrected_em`:
- `original = 0.90075`
- `img_black = 0.57575`
- `img_blur_20 = 0.58575`
- `img_blur_50 = 0.57575`

### Field-type extraction in `v2`

Для сценария `seen img_black`:

Florence:
- `AMOUNT = 0.169118`
- `CONTACT_ADR = 0.051471`
- `DATE = 0.086364`
- `ID = 0.045455`
- `ORG = 0.071212`
- `PERSON = 0.059091`

Qwen-2B:
- `AMOUNT = 0.404412`
- `CONTACT_ADR = 0.404412`
- `DATE = 0.450000`
- `ID = 0.409091`
- `ORG = 0.465152`
- `PERSON = 0.725758`

Qwen-7B:
- `AMOUNT = 0.463235`
- `CONTACT_ADR = 0.522059`
- `DATE = 0.540909`
- `ID = 0.560606`
- `ORG = 0.556061`
- `PERSON = 0.816667`

### Mode comparison in `v2`

Средний `seen corrected_em`:
- `Qwen-2B image-only = 0.56325`
- `Qwen-2B image+OCR = 0.16575`
- `Qwen-7B image-only = 0.65950`

### Interpretation `v2` Extraction

- extraction leakage сохраняется и в controlled setup;
- `Qwen-7B` выглядит strongest model по image-only extraction;
- `PERSON` — наиболее уязвимый coarse type в masked extraction;
- OCR-канал у `Qwen-2B` снова существенно слабее visual channel;
- сильный blur почти не убивает extraction у Qwen, что косвенно указывает на memorized internal associations, а не просто на распознавание видимого текста.

## 15. Final Comparative Interpretation

### 15.1 Что устойчиво подтверждается и в `v1`, и в `v2`

1. Fine-tuning приводит к privacy leakage.
2. Leakage проявляется через две независимые линии:
   - `loss-based MIA`;
   - `data extraction`.
3. Baseline модели текут существенно слабее.
4. Leakage заметно выше на `seen`, чем на `unseen`.
5. Разные модели проявляют leakage по-разному:
   - Florence особенно сильна по `MIA loss`;
   - Qwen сильнее в extraction под masked image scenarios.

### 15.2 Что особенно интересно

- `v1` показывает privacy risk в practical setup.
- `v2` показывает, что эффект не исчезает даже при выровненном budget.
- Это делает выводы сильнее, чем если бы был только один эксперимент.

### 15.3 Что нельзя утверждать слишком сильно

Нельзя честно утверждать:
- что размер модели сам по себе полностью объясняет разницу в leakage;
- что `confidence-based MIA` у Qwen надежно измерен.

Причины:
- в `v1` training regimes различаются;
- в `v1` и в сохраненных outputs `v2` confidence для Qwen остается технически ненадежным.

## 16. Limitations

Главные ограничения проекта:

1. `v1` не является идеально симметричным benchmark:
   - разные training budgets;
   - разные режимы fine-tuning;
   - разный объем обучающих данных у Florence и Qwen.

2. `confidence-based MIA` для Qwen не удалось довести до надежного состояния в сохраненных outputs:
   - поэтому основной MIA-вывод строится по `loss`.

3. Florence и Qwen относятся к разным семействам моделей:
   - их нельзя трактовать как differ only in parameter count.

4. OCR-ветка полноценно исследована только для `Qwen-2B`:
   - это означает, что image+OCR comparison ограничен.

Эти ограничения не отменяют результаты, но влияют на то, как именно их нужно формулировать в тексте.

## 17. Storage and Artifacts

### 17.1 Repository

В git-репозитории лежат:
- код;
- ноутбуки;
- небольшие артефакты;
- README;
- вспомогательные CSV и manifests.

### 17.2 Hugging Face

Используется для:
- исходного датасета;
- загрузки и хранения checkpoints моделей.

Ветки checkpoint-ов разделены между `v1` и `v2`.

### 17.3 Google Drive

Используется как основное хранилище итоговых артефактов экспериментов.

#### `v1`

Основной корень:
- `MyDrive/course_work2026/artifacts/`

Ключевые подпапки:
- `finetuning_generative/`
- `privacy_attacks/mia/`
- `privacy_attacks/extraction/`
- `analysis/`

#### `v2`

Отдельный корень:
- `MyDrive/course_work2026/artifacts_v2_controlled/`

Ключевые подпапки:
- `finetuning_generative/`
- `privacy_attacks/mia/`
- `privacy_attacks/extraction/`
- `analysis/`

Это сделано специально, чтобы `v2` не перетирал `v1`.

### 17.4 Comet

Используется для:
- логирования train metrics;
- хранения figures;
- хранения dataframes;
- просмотра runs по ноутбукам.

Но итоговым source of truth для `22-24` служат прежде всего CSV и figures в Google Drive.

## 18. What to Use in the Thesis

Если писать по проекту текст курсовой, то удобно использовать такой принцип.

### Для методологии

Использовать разделы:
- dataset;
- field labeling;
- benchmark construction;
- models;
- `v1` and `v2` setup;
- metrics;
- redaction scenarios.

### Для результатов

Основные артефакты:
- `24_final_analysis.ipynb`
- `24_final_analysis_v2.ipynb`
- итоговые CSV и figures из папок `analysis/`

### Для главных количественных выводов

Использовать:
- `loss-based MIA`
- extraction `corrected_em`
- field-type analysis
- mode comparison
- qualitative successful extraction examples

### Что лучше не делать опорой текста

Не стоит делать основным доказательством:
- `Qwen confidence-based MIA`

## 19. Recommended Narrative for the Thesis

Если строить текст курсовой по результатам проекта, логика может быть такой:

1. В document AI скрытые поля могут утекать после fine-tuning.
2. Для проверки этого строится benchmark с типизацией полей и controlled redaction scenarios.
3. Модели дообучаются в двух режимах:
   - practical (`v1`);
   - controlled (`v2`).
4. В обоих режимах после fine-tuning возникает сильный `loss-based MIA`.
5. В обоих режимах fine-tuned модели способны восстанавливать скрытые значения на `seen` документах.
6. Leakage зависит от:
   - архитектуры;
   - режима обучения;
   - типа поля;
   - сценария redaction.
7. Наиболее сильный visual extraction leakage в controlled setup показывает `Qwen-7B`, а strongest `MIA loss` — Florence.

## 20. Bottom Line

Краткий честный итог проекта:

- работа не является идеально чистым benchmark paper;
- но это сильное прикладное исследование с реальными, устойчивыми privacy-signal результатами;
- наличие `v1` и `v2` делает выводы заметно убедительнее;
- основная исследовательская ценность проекта в том, что privacy leakage был показан:
  - на нескольких моделях;
  - через две разные атаки;
  - на реалистичных документных данных;
  - с анализом по coarse field types и сценариям редактирования.

Если нужен самый короткий вывод в одну фразу:

> Fine-tuning генеративных document VLM на seen документах приводит к заметному privacy leakage, которое проявляется и через loss-based membership inference, и через extraction скрытых полей, а устойчивость этого вывода подтверждается как в practical, так и в controlled setup.
