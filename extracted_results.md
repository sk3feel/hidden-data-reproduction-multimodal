# Curated Experimental Results

This file is a reduced, thesis-oriented version of the raw extraction dump.

Included here:
- the field labeling results that are most likely useful for the methodology section;
- the final experimental notebooks:
  - `20-24`
  - `20-24_v2`
- only the outputs and tables that are most likely useful in the thesis text.

Excluded from this file:
- project tree;
- service scripts;
- helper utilities;
- low-level implementation details that are unlikely to be cited in the thesis;
- noisy intermediate outputs that do not add methodological or experimental value.

## Supporting Material Likely Useful for the Thesis

### `src/docqa_benchmark.py`

**Description:** Benchmark construction and coarse field type mapping used by the experimental notebooks.

**Useful mappings from code:**

| Source field type | Coarse field type | Source |
|---|---|---|
| `DATE_TIME` | `DATE` | code |
| `MONEY` | `AMOUNT` | code |
| `QUANTITY` | `AMOUNT` | code |
| `PERCENTAGE` | `AMOUNT` | code |
| `IDENTIFIER` | `ID` | code |
| `DOCUMENT_REFERENCE` | `ID` | code |
| `PERSON_NAME` | `PERSON` | code |
| `ORG_NAME` | `ORG` | code |
| `ADDRESS` | `CONTACT_ADR` | code |
| `CONTACT` | `CONTACT_ADR` | code |

**Manifest fields written by code:**
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

### `src/docqa_metrics.py`

**Description:** Metric definitions used by extraction notebooks.

**Useful metric definitions from code:**
- `best_metric_over_answers(prediction, answers)` returns the best `exact_match` and `token_f1` over all acceptable answers.
- `build_answer_pool(records)` builds answer pools grouped by `coarse_field_type`.
- `estimate_random_baseline(record, answer_pool)` estimates a random baseline from answers of the same coarse field type.

## Field Labeling and Type Distribution

### `notebooks/03_gigachat_field_labeling.ipynb`

**Description:** Pilot field labeling on a smaller subset of DocVQA question-answer pairs.

**Useful parameters:**
- `OUTPUT_CSV = PROJECT_ROOT / "docvqa_field_types_pilot.csv"`
- closed type list:
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

**Results:**

| Metric | Value | Source |
|---|---:|---|
| Output CSV path | `C:\Users\scfeel\DEV\hse\3year-2sem\course_work2026\docvqa_field_types_pilot.csv` | output |
| Saved rows | `200` | `Saved 200 rows to: ...` |
| OTHER examples | `4` | output |

**Table from output: field type counts**

```text
field_type  count
0        AMOUNT     62
2           ORG     21
3          DATE     18
4            ID     16
6     REFERENCE      9
7       ADDRESS      7
8    PERCENTAGE      6
9         PHONE      6
10        OTHER      4
```

### `notebooks/04_gigachat_field_labeling_scalable.ipynb`

**Description:** Full scalable field labeling pipeline with rules + LLM and evaluation against a gold subset.

**Useful parameters from code:**
- label source can be `rule` or `llm`
- additional metadata columns:
  - `field_group`
  - `sensitivity`
  - `label_source`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| OTHER examples | `39` | output |
| accuracy | `0.8367` | classification report output |
| macro avg precision | `0.8385` | classification report output |
| macro avg recall | `0.8338` | classification report output |
| macro avg f1-score | `0.8264` | classification report output |
| weighted avg precision | `0.8510` | classification report output |
| weighted avg recall | `0.8367` | classification report output |
| weighted avg f1-score | `0.8353` | classification report output |
| support | `300` | classification report output |
| rule accuracy | `0.8788` | `rule: n=132, accuracy=0.8788` |
| rule n | `132` | same line |
| llm accuracy | `0.8036` | `llm: n=168, accuracy=0.8036` |
| llm n | `168` | same line |
| confusion matrix CSV | `C:\Users\scfeel\DEV\hse\3year-2sem\course_work2026\artifacts\field_labeling\gold_annotation\evaluation\pixparse__docvqa-single-page-questions__validation__gold_confusion_matrix.csv` | output |

**Table from output: field type counts**

```text
field_type  count
0          PERSON_NAME    650
1            DATE_TIME    639
5           IDENTIFIER    387
6             ORG_NAME    382
8   DOCUMENT_REFERENCE    231
9           PERCENTAGE    213
15               OTHER     39
16             ADDRESS     36
```

**Table from output: classification report**

```text
                    precision    recall  f1-score   support
...
          accuracy                         0.8367       300
         macro avg     0.8385    0.8338    0.8264       300
      weighted avg     0.8510    0.8367    0.8353       300
```

## Main Experiments: Version 1

### `notebooks/20_florence2_finetune.ipynb`

**Description:** Fine-tuning `microsoft/Florence-2-base` for generative document VQA.

**Key parameters:**
- model: `microsoft/Florence-2-base`
- final checkpoint used later: `epoch_30`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| final seen EM | `0.96` | output |
| final unseen EM | `0.1` | output |

**Output snippet**

```text
{'seen_em': 0.96, 'unseen_em': 0.1}
```

### `notebooks/21_qwen2vl_finetune.ipynb`

**Description:** Fine-tuning `Qwen2-VL-2B` and `Qwen2-VL-7B` with QLoRA.

**Key parameters:**
- models:
  - `Qwen/Qwen2-VL-2B-Instruct`
  - `Qwen/Qwen2-VL-7B-Instruct`
- final checkpoints used later:
  - `qwen2b -> epoch_20`
  - `qwen7b -> epoch_10`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| Qwen-2B Sanity EM@5 | `0.30` | output |
| Qwen-2B Sanity EM@10 | `0.35` | output |
| Qwen-2B Sanity EM@20 | `0.80` | output |
| Qwen-2B final seen EM | `0.92` | output |
| Qwen-2B final unseen EM | `0.16` | output |
| Qwen-7B Sanity EM@5 | `0.30` | output |
| Qwen-7B Sanity EM@10 | `0.65` | output |
| Qwen-7B final seen EM | `0.80` | output |
| Qwen-7B final unseen EM | `0.21` | output |

**Output snippets**

```text
  Sanity EM@5: 0.30
  Sanity EM@10: 0.35
  Sanity EM@20: 0.80
qwen2b: seen_em=0.92, unseen_em=0.16
```

```text
  Sanity EM@5: 0.30
  Sanity EM@10: 0.65
qwen7b: seen_em=0.80, unseen_em=0.21
```

### `notebooks/22_membership_inference.ipynb`

**Description:** Membership inference attack for baseline and fine-tuned models.

**Key parameters:**
- baseline models:
  - Florence-2 base
  - Qwen2-VL-2B base
- fine-tuned checkpoints:
  - `florence2 -> epoch_30`
  - `qwen2b -> epoch_20`
  - `qwen7b -> epoch_10`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| qwen2b sanity seen EM | `0.93` | output |
| qwen2b sanity unseen EM | `0.12` | output |
| qwen7b sanity seen EM | `0.8` | output |
| qwen7b sanity unseen EM | `0.19` | output |
| baseline_florence2_auc_confidence | `0.572125` | Comet summary output |
| baseline_florence2_auc_loss | `0.51395` | Comet summary output |
| baseline_qwen2b_auc_confidence | `0.5` | Comet summary output |
| baseline_qwen2b_auc_loss | `0.5492250000000001` | Comet summary output |
| florence2_epoch30_auc_confidence | `0.5777875` | Comet summary output |
| florence2_epoch30_auc_loss | `0.9920000000000001` | Comet summary output |
| qwen2b_epoch20_auc_confidence | `0.5` | Comet summary output |
| qwen2b_epoch20_auc_loss | `0.963775` | Comet summary output |
| qwen7b_epoch10_auc_confidence | `0.5` | Comet summary output |
| qwen7b_epoch10_auc_loss | `0.939875` | Comet summary output |
| saved CSV count | `9` | output |
| output directory | `/content/drive/MyDrive/course_work2026/artifacts/privacy_attacks/mia` | output |

**Table from output: baseline summary**

```text
tag  auc_confidence  auc_loss    t_stat  p_value  mean_seen_conf  ...
```

**Table from output: fine-tuned summary**

```text
tag checkpoint  auc_confidence  auc_loss    t_stat   p_value
0  florence2   epoch_30        0.577788  0.992000  3.443347  0.000642
1     qwen2b   epoch_20        0.500000  0.963775       NaN       NaN
2     qwen7b   epoch_10        0.500000  0.939875       NaN       NaN
```

### `notebooks/23_data_extraction.ipynb`

**Description:** Data extraction attacks under image-only and image+OCR settings.

**Key parameters:**
- Florence baseline and `epoch_30`
- Qwen-2B `epoch_20`
- Qwen-7B `epoch_10`
- image-only scenarios:
  - `original`
  - `img_black`
  - `img_blur_20`
  - `img_blur_50`
- image+OCR scenarios for Qwen-2B

**Results from Comet outputs:**

| Metric | Value | Source |
|---|---:|---|
| baseline seen img_black corrected_em | `-0.01425` | Comet summary output |
| baseline seen img_blur_20 corrected_em | `-0.01425` | Comet summary output |
| baseline seen img_blur_50 corrected_em | `-0.01425` | Comet summary output |
| baseline seen original corrected_em | `-0.01425` | Comet summary output |
| florence2 seen img_black corrected_em | `0.14575` | Comet summary output |
| florence2 seen img_blur_20 corrected_em | `0.15075000000000002` | Comet summary output |
| florence2 seen img_blur_50 corrected_em | `0.15575` | Comet summary output |
| florence2 seen original corrected_em | `0.9657499999999999` | Comet summary output |
| qwen2b image-only seen img_black corrected_em | `0.59575` | Comet summary output |
| qwen2b image-only seen img_blur_20 corrected_em | `0.60075` | Comet summary output |
| qwen2b image-only seen img_blur_50 corrected_em | `0.60075` | Comet summary output |
| qwen2b image-only seen original corrected_em | `0.94575` | Comet summary output |
| qwen2b image+OCR seen img_black__ocr_mask__k_0 corrected_em | `0.02575` | Comet summary output |
| qwen2b image+OCR seen img_none__ocr_mask__k_0 corrected_em | `0.32075000000000004` | Comet summary output |

**Useful table from output: all-scenario image-only metrics**

```text
split  scenario     coarse_type  n_examples  exact_match  token_f1  random_em  random_f1  corrected_em  corrected_f1  tag
seen   img_black    ALL          200         ...
seen   img_blur_20  ALL          200         ...
seen   img_blur_50  ALL          200         ...
seen   original     ALL          200         ...
```

### `notebooks/24_final_analysis.ipynb`

**Description:** Final aggregation notebook for version 1.

**Results:**

| Metric | Value | Source |
|---|---:|---|
| mia_csv_count | `9` | output |
| extraction_csv_count | `21` | output |
| train_loss_csv_count | `0` | output |

**Output snippet**

```text
{'mia_csv_count': 9, 'extraction_csv_count': 21, 'train_loss_csv_count': 0}
```

**Useful tables from output**

```text
tag  auc_confidence  auc_loss      params  log_params  model_label
0  florence2  0.577788  0.992000   230000000  8.361728  Florence-2-base
1     qwen2b  0.500000  0.963775  2000000000  9.301030  Qwen2-VL-2B
2     qwen7b  0.500000  0.939875  7000000000  9.845098  Qwen2-VL-7B
```

```text
tag      model_label   mia_auc  extraction_corrected_em
florence2  Florence-2-base  0.992000  0.14575
qwen2b     Qwen2-VL-2B      0.963775  0.59575
qwen7b     Qwen2-VL-7B      0.939875  0.54575
```

## Main Experiments: Version 2

### `notebooks/20_florence2_finetune_v2.ipynb`

**Description:** Controlled rerun of Florence-2 fine-tuning on the shared balanced subset.

**Key parameters:**
- final checkpoint used later: `epoch_10`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| final seen EM | `0.9` | output |
| final unseen EM | `0.08` | output |

**Output snippet**

```text
{'seen_em': 0.9, 'unseen_em': 0.08}
```

### `notebooks/21_qwen2vl_finetune_v2.ipynb`

**Description:** Controlled rerun of Qwen2-VL-2B and Qwen2-VL-7B fine-tuning on the shared balanced subset.

**Key parameters:**
- final checkpoints used later:
  - `qwen2b -> epoch_10`
  - `qwen7b -> epoch_10`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| Qwen-2B Sanity EM@3 | `0.20` | output |
| Qwen-2B Sanity EM@5 | `0.30` | output |
| Qwen-2B Sanity EM@10 | `0.55` | output |
| Qwen-2B final seen EM | `0.77` | output |
| Qwen-2B final unseen EM | `0.20` | output |
| Qwen-7B Sanity EM@3 | `0.25` | output |
| Qwen-7B Sanity EM@5 | `0.35` | output |
| Qwen-7B Sanity EM@10 | `0.60` | output |
| Qwen-7B final seen EM | `0.84` | output |
| Qwen-7B final unseen EM | `0.24` | output |

**Output snippets**

```text
  Sanity EM@3: 0.20
  Sanity EM@5: 0.30
  Sanity EM@10: 0.55
qwen2b: seen_em=0.77, unseen_em=0.20
```

```text
  Sanity EM@3: 0.25
  Sanity EM@5: 0.35
  Sanity EM@10: 0.60
qwen7b: seen_em=0.84, unseen_em=0.24
```

### `notebooks/22_membership_inference_v2.ipynb`

**Description:** Controlled rerun of membership inference on the balanced seen/unseen split.

**Key parameters:**
- fine-tuned checkpoints:
  - `florence2 -> epoch_10`
  - `qwen2b -> epoch_10`
  - `qwen7b -> epoch_10`

**Results:**

| Metric | Value | Source |
|---|---:|---|
| baseline_florence2_auc_confidence | `0.5786875` | Comet summary output |
| baseline_florence2_auc_loss | `0.5169750000000001` | Comet summary output |
| baseline_qwen2b_auc_confidence | `0.5` | Comet summary output |
| baseline_qwen2b_auc_loss | `0.5493` | Comet summary output |
| florence2_epoch10_auc_confidence | `0.58275` | Comet summary output |
| florence2_epoch10_auc_loss | `0.988675` | Comet summary output |
| qwen2b_epoch10_auc_confidence | `0.5` | Comet summary output |
| qwen2b_epoch10_auc_loss | `0.9207000000000001` | Comet summary output |
| qwen7b_epoch10_auc_confidence | `0.5` | Comet summary output |
| qwen7b_epoch10_auc_loss | `0.941725` | Comet summary output |
| saved CSV count | `9` | output |
| output directory | `/content/drive/MyDrive/course_work2026/artifacts_v2_controlled/privacy_attacks/mia` | output |

**Table from output: fine-tuned summary**

```text
tag checkpoint  auc_confidence  auc_loss    t_stat   p_value
...
```

### `notebooks/23_data_extraction_v2.ipynb`

**Description:** Controlled rerun of extraction attacks on the balanced split.

**Results used later in final analysis (`24_v2`):**

| Metric | Value | Source |
|---|---:|---|
| Florence image-only seen original corrected_em | `0.93075` | final analysis output derived from extraction CSVs |
| Florence image-only seen img_black corrected_em | `0.08075` | final analysis output derived from extraction CSVs |
| Qwen-2B image-only seen original corrected_em | `0.80575` | final analysis output derived from extraction CSVs |
| Qwen-2B image-only seen img_black corrected_em | `0.47575` | final analysis output derived from extraction CSVs |
| Qwen-7B image-only seen original corrected_em | `0.90075` | final analysis output derived from extraction CSVs |
| Qwen-7B image-only seen img_black corrected_em | `0.57575` | final analysis output derived from extraction CSVs |
| Qwen-2B image+OCR seen img_black__ocr_mask__k_0 corrected_em | `0.01575` | final analysis output derived from extraction CSVs |
| Qwen-2B image+OCR seen img_none__ocr_mask__k_0 corrected_em | `0.31575` | final analysis output derived from extraction CSVs |

### `notebooks/24_final_analysis_v2.ipynb`

**Description:** Final aggregation notebook for version 2.

**Useful counts from output:**

| Metric | Value | Source |
|---|---:|---|
| mia_csv_count | `9` | output |
| extraction_csv_count | `21` | output |
| train_loss_csv_count | `0` | output |

**Useful tables from output**

```text
tag  auc_confidence  auc_loss      params  log_params  model_label
0  florence2  0.582750  0.988675   230000000  8.361728  Florence-2-base
1     qwen2b  0.500000  0.920700  2000000000  9.301030  Qwen2-VL-2B
2     qwen7b  0.500000  0.941725  7000000000  9.845098  Qwen2-VL-7B
```

```text
split  scenario     coarse_type  n_examples  exact_match  token_f1  random_em  random_f1  corrected_em  corrected_f1  tag  model_label
0  seen  img_black  ALL  200  ...  0.08075  ...  florence2  Florence-2-base
4  seen  img_black  ALL  200  ...  0.47575  ...  qwen2b     Qwen2-VL-2B
8  seen  img_black  ALL  200  ...  0.57575  ...  qwen7b     Qwen2-VL-7B
```

```text
split  scenario                  coarse_type  n_examples  exact_match  token_f1  random_em  random_f1  corrected_em  corrected_f1  tag  model_label
0  seen  img_black__ocr_mask__k_0  ALL  200  ...  0.01575  ...  qwen2b  Qwen2-VL-2B
1  seen  img_none__ocr_mask__k_0   ALL  200  ...  0.31575  ...  qwen2b  Qwen2-VL-2B
```

```text
model_label  coarse_type  corrected_em  corrected_f1  exact_match  token_f1
...
```

```text
tag  model_label        mode       mean_corrected_em
0  qwen2b  Qwen2-VL-2B  image_only  0.56325
1  qwen2b  Qwen2-VL-2B  image+OCR   0.16575
2  qwen7b  Qwen2-VL-7B  image_only  0.65950
```

```text
tag      model_label   mia_auc  extraction_corrected_em
0  florence2  Florence-2-base  0.988675  0.08075
1  qwen2b     Qwen2-VL-2B      0.920700  0.47575
2  qwen7b     Qwen2-VL-7B      0.941725  0.57575
```

## Notes

- Values above were kept only if they are likely to be useful for the thesis text.
- When a number is taken from `24` or `24_v2`, it is copied from the final analysis notebook output or from tables displayed there.
- This file is intentionally curated and does not aim to preserve every single intermediate output from the repository.

## Final Notebook Graphs

The entries below describe graphs from:
- `notebooks/24_final_analysis.ipynb`
- `notebooks/24_final_analysis_v2.ipynb`

The numbers are taken only from code-visible data sources or from tables already printed in notebook outputs.

### Graph: `AUC-ROC по методу confidence (средняя log-prob)` and `AUC-ROC по методу loss (cross-entropy)` (`24_final_analysis.ipynb`)
- File/cell: cell 8; saved as `artifacts/analysis/figures/mia_bar_baseline_vs_finetuned.png`
- X axis: `Модель` / model label
- Y axis: `AUC-ROC`
- Data series: `stage = baseline`, `stage = fine-tuned`
- Key values:
  - Data coincide with the MIA tables already listed above for `notebooks/22_membership_inference.ipynb`
  - confidence:
    - Florence-2 baseline = `0.572125`
    - Florence-2 fine-tuned = `0.5777875`
    - Qwen2-VL-2B baseline = `0.5`
    - Qwen2-VL-2B fine-tuned = `0.5`
    - Qwen2-VL-7B fine-tuned = `0.5`
  - loss:
    - Florence-2 baseline = `0.51395`
    - Florence-2 fine-tuned = `0.9920000000000001`
    - Qwen2-VL-2B baseline = `0.5492250000000001`
    - Qwen2-VL-2B fine-tuned = `0.963775`
    - Qwen2-VL-7B fine-tuned = `0.939875`

### Graph: `mia_confidence_histograms` (`24_final_analysis.ipynb`)
- File/cell: cell 9; saved as `artifacts/analysis/figures/mia_confidence_histograms.png`
- X axis: `Средняя log-prob ответа`
- Y axis: `Частота`
- Data series: one panel per model label
  - `Florence-2-base`
  - `Qwen2-VL-2B`
  - `Qwen2-VL-7B`
- Key values:
  - [data not extracted, rerun cell with explicit print of histogram source tables is needed]

### Graph: `AUC-ROC MIA по типам полей` (`24_final_analysis.ipynb`)
- File/cell: cell 10; saved as `artifacts/analysis/figures/mia_heatmap_model_fieldtype_auc.png`
- X axis: `Тип поля`
- Y axis: `Модель`
- Data series: heatmap values by `model_label × coarse_type`
- Key values:
  - Data coincide with the field-type MIA table already listed above
  - The plotted heatmap uses `auc_confidence`
  - Printed output table also contains `auc_loss` and `n_examples`

### Graph: `Кривая MIA по эпохам` (`24_final_analysis.ipynb`)
- File/cell: cell 11; intended save path `artifacts/analysis/figures/mia_epoch_curve.png`
- X axis: `Эпоха`
- Y axis: `AUC-ROC`
- Data series: one line per model label
- Key values:
  - [data not extracted, notebook output explicitly says `mia_epoch_curve.csv not found; skipping epoch curve plot.`]

### Graph: `Кривая train loss по эпохам` (`24_final_analysis.ipynb`)
- File/cell: cell 11; intended save path `artifacts/analysis/figures/training_loss_curves.png`
- X axis: `Эпоха`
- Y axis: `Train loss`
- Data series: one line per training run
- Key values:
  - [data not extracted, notebook output explicitly says `Training loss CSV files not found; skipping train loss plot.`]

### Graph: `Scaling MIA: log(параметры модели) -> AUC-ROC` (`24_final_analysis.ipynb`)
- File/cell: cell 12; saved as `artifacts/analysis/figures/mia_scaling_auc.png`
- X axis: `log10(число параметров)`
- Y axis: `AUC-ROC`
- Data series:
  - `confidence`
  - `loss`
- Key values:
  - Data coincide with the scaling table already listed above
  - Florence-2:
    - `log_params = 8.361728`
    - `auc_confidence = 0.577788`
    - `auc_loss = 0.992000`
  - Qwen2-VL-2B:
    - `log_params = 9.301030`
    - `auc_confidence = 0.500000`
    - `auc_loss = 0.963775`
  - Qwen2-VL-7B:
    - `log_params = 9.845098`
    - `auc_confidence = 0.500000`
    - `auc_loss = 0.939875`

### Graph: `Скорректированный EM по image-only сценариям (seen)` (`24_final_analysis.ipynb`)
- File/cell: cell 14; saved as `artifacts/analysis/figures/extraction_heatmap_image_only_seen.png`
- X axis: `Image-сценарий`
- Y axis: `Модель`
- Data series: heatmap values by `model_label × scenario`
- Key values:
  - Data coincide with the image-only extraction table already listed above
  - Explicit values available from outputs:
    - Florence-2-base:
      - `img_black = 0.14575`
      - `img_blur_20 = 0.15075000000000002`
      - `img_blur_50 = 0.15575`
      - `original = 0.9657499999999999`
    - Qwen2-VL-2B:
      - `img_black = 0.59575`
      - `img_blur_20 = 0.60075`
      - `img_blur_50 = 0.60075`
      - `original = 0.94575`
    - Qwen2-VL-7B:
      - `img_black = 0.54575`
      - `img_blur_20 = 0.54575`
      - `img_blur_50 = 0.55075`
      - `original = 0.88075`

### Graph: `Скорректированный EM для Qwen: image+OCR сценарии (seen)` (`24_final_analysis.ipynb`)
- File/cell: cell 15; saved as `artifacts/analysis/figures/extraction_heatmap_image_ocr_seen.png`
- X axis: `Сценарий`
- Y axis: `Модель`
- Data series: heatmap values for Qwen image+OCR scenarios
- Key values:
  - Data coincide with the image+OCR extraction table already listed above
  - Explicit values available from outputs:
    - Qwen2-VL-2B:
      - `img_black__ocr_mask__k_0 = 0.02575`
      - `img_none__ocr_mask__k_0 = 0.32075000000000004`

### Graph: `Скорректированный EM по типам полей, сценарий img_black (seen)` (`24_final_analysis.ipynb`)
- File/cell: cell 16; saved as `artifacts/analysis/figures/extraction_bar_fieldtype_img_black.png`
- X axis: `Тип поля`
- Y axis: `Скорректированный EM`
- Data series: one bar group per `model_label`
- Key values:
  - Data coincide with the field-type extraction table already listed above
  - This graph uses `field_type_df`
  - The printed table contains:
    - `coarse_type`
    - `corrected_em`
    - `corrected_f1`
    - `exact_match`
    - `token_f1`

### Graph: `Влияние blur sigma на скорректированный EM (seen)` (`24_final_analysis.ipynb`)
- File/cell: cell 17; saved as `artifacts/analysis/figures/extraction_blur_sigma_curve.png`
- X axis: `Sigma blur`
- Y axis: `Скорректированный EM`
- Data series:
  - `Florence-2-base`
  - `Qwen2-VL-2B`
  - `Qwen2-VL-7B`
- Key values:
  - Florence-2-base:
    - `sigma 20 -> 0.15075000000000002`
    - `sigma 50 -> 0.15575`
  - Qwen2-VL-2B:
    - `sigma 20 -> 0.60075`
    - `sigma 50 -> 0.60075`
  - Qwen2-VL-7B:
    - `sigma 20 -> 0.54575`
    - `sigma 50 -> 0.55075`

### Graph: `Сравнение режимов: image-only vs image+OCR` (`24_final_analysis.ipynb`)
- File/cell: cell 18; saved as `artifacts/analysis/figures/extraction_mode_compare.png`
- X axis: `Модель`
- Y axis: `Средний скорректированный EM`
- Data series:
  - `image_only`
  - `image+OCR`
- Key values:
  - Data coincide with the mode comparison table already listed above
  - Explicit values available from outputs:
    - Qwen2-VL-2B `image_only = 0.68575`
    - Qwen2-VL-2B `image+OCR = 0.17325`
    - Qwen2-VL-7B `image_only = 0.63075`

### Graph: `Кривая extraction по эпохам (img_black, seen)` (`24_final_analysis.ipynb`)
- File/cell: cell 19; intended save path `artifacts/analysis/figures/extraction_epoch_curve.png`
- X axis: `Эпоха`
- Y axis: `Скорректированный EM`
- Data series: one line per model label
- Key values:
  - [data not extracted, notebook output indicates epoch-curve data were not available]

### Graph: `Связь MIA и extraction` (`24_final_analysis.ipynb`)
- File/cell: cell 20; saved as `artifacts/analysis/figures/mia_vs_extraction_scatter.png`
- X axis: `MIA AUC-ROC по loss`
- Y axis: `Extraction скорректированный EM (img_black, seen)`
- Data series: one point per model
- Key values:
  - Data coincide with the scatter table already listed above
  - Florence-2-base:
    - `mia_auc = 0.992000`
    - `extraction_corrected_em = 0.14575`
  - Qwen2-VL-2B:
    - `mia_auc = 0.963775`
    - `extraction_corrected_em = 0.59575`
  - Qwen2-VL-7B:
    - `mia_auc = 0.939875`
    - `extraction_corrected_em = 0.54575`

### Graph: `qualitative_successful_extractions` (`24_final_analysis.ipynb`)
- File/cell: cell 23; saved as `artifacts/analysis/figures/qualitative_successful_extractions.png`
- X axis: none
- Y axis: none
- Data series: image grid of successful extraction examples
- Key values:
  - [this is a qualitative figure; no numeric axes]

### Graph: `AUC-ROC по методу confidence (средняя log-prob)` and `AUC-ROC по методу loss (cross-entropy)` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 8; saved as `artifacts_v2_controlled/analysis/figures/mia_bar_baseline_vs_finetuned.png`
- X axis: `Модель` / model label
- Y axis: `AUC-ROC`
- Data series: `stage = baseline`, `stage = fine-tuned`
- Key values:
  - Data coincide with the MIA tables already listed above for `notebooks/22_membership_inference_v2.ipynb`
  - confidence:
    - Florence-2 baseline = `0.5786875`
    - Florence-2 fine-tuned = `0.58275`
    - Qwen2-VL-2B baseline = `0.5`
    - Qwen2-VL-2B fine-tuned = `0.5`
    - Qwen2-VL-7B fine-tuned = `0.5`
  - loss:
    - Florence-2 baseline = `0.5169750000000001`
    - Florence-2 fine-tuned = `0.988675`
    - Qwen2-VL-2B baseline = `0.5493`
    - Qwen2-VL-2B fine-tuned = `0.9207000000000001`
    - Qwen2-VL-7B fine-tuned = `0.941725`

### Graph: `mia_confidence_histograms` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 9; saved as `artifacts_v2_controlled/analysis/figures/mia_confidence_histograms.png`
- X axis: `Средняя log-prob ответа`
- Y axis: `Частота`
- Data series: one panel per model label
  - `Florence-2-base`
  - `Qwen2-VL-2B`
  - `Qwen2-VL-7B`
- Key values:
  - [data not extracted, rerun cell with explicit print of histogram source tables is needed]

### Graph: `AUC-ROC MIA по типам полей` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 10; saved as `artifacts_v2_controlled/analysis/figures/mia_heatmap_model_fieldtype_auc.png`
- X axis: `Тип поля`
- Y axis: `Модель`
- Data series: heatmap values by `model_label × coarse_type`
- Key values:
  - Data coincide with the field-type MIA table already listed above for `v2`
  - The plotted heatmap uses `auc_confidence`
  - Printed output table also contains `auc_loss` and `n_examples`

### Graph: `Кривая MIA по эпохам` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 11; intended save path `artifacts_v2_controlled/analysis/figures/mia_epoch_curve.png`
- X axis: `Эпоха`
- Y axis: `AUC-ROC`
- Data series: one line per model label
- Key values:
  - [data not extracted, notebook output says the epoch-curve file was not found]

### Graph: `Кривая train loss по эпохам` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 11; intended save path `artifacts_v2_controlled/analysis/figures/training_loss_curves.png`
- X axis: `Эпоха`
- Y axis: `Train loss`
- Data series: one line per training run
- Key values:
  - [data not extracted, notebook output says the train-loss CSV files were not found]

### Graph: `Scaling MIA: log(параметры модели) -> AUC-ROC` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 12; saved as `artifacts_v2_controlled/analysis/figures/mia_scaling_auc.png`
- X axis: `log10(число параметров)`
- Y axis: `AUC-ROC`
- Data series:
  - `confidence`
  - `loss`
- Key values:
  - Data coincide with the scaling table already listed above
  - Florence-2:
    - `log_params = 8.361728`
    - `auc_confidence = 0.582750`
    - `auc_loss = 0.988675`
  - Qwen2-VL-2B:
    - `log_params = 9.301030`
    - `auc_confidence = 0.500000`
    - `auc_loss = 0.920700`
  - Qwen2-VL-7B:
    - `log_params = 9.845098`
    - `auc_confidence = 0.500000`
    - `auc_loss = 0.941725`

### Graph: `Скорректированный EM по image-only сценариям (seen)` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 14; saved as `artifacts_v2_controlled/analysis/figures/extraction_heatmap_image_only_seen.png`
- X axis: `Image-сценарий`
- Y axis: `Модель`
- Data series: heatmap values by `model_label × scenario`
- Key values:
  - Data coincide with the image-only extraction table already listed above for `v2`
  - Explicit values available from outputs:
    - Florence-2-base:
      - `original = 0.93075`
      - `img_black = 0.08075`
      - `img_blur_20 = 0.08075`
      - `img_blur_50 = 0.08075`
    - Qwen2-VL-2B:
      - `original = 0.80575`
      - `img_black = 0.47575`
      - `img_blur_20 = 0.48075`
      - `img_blur_50 = 0.49075`
    - Qwen2-VL-7B:
      - `original = 0.90075`
      - `img_black = 0.57575`
      - `img_blur_20 = 0.58575`
      - `img_blur_50 = 0.57575`

### Graph: `Скорректированный EM для Qwen: image+OCR сценарии (seen)` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 15; saved as `artifacts_v2_controlled/analysis/figures/extraction_heatmap_image_ocr_seen.png`
- X axis: `Сценарий`
- Y axis: `Модель`
- Data series: heatmap values for Qwen image+OCR scenarios
- Key values:
  - Data coincide with the image+OCR extraction table already listed above for `v2`
  - Explicit values available from outputs:
    - Qwen2-VL-2B:
      - `img_black__ocr_mask__k_0 = 0.01575`
      - `img_none__ocr_mask__k_0 = 0.31575`

### Graph: `Скорректированный EM по типам полей, сценарий img_black (seen)` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 16; saved as `artifacts_v2_controlled/analysis/figures/extraction_bar_fieldtype_img_black.png`
- X axis: `Тип поля`
- Y axis: `Скорректированный EM`
- Data series: one bar group per `model_label`
- Key values:
  - Data coincide with the field-type extraction table already listed above for `v2`
  - The printed table contains:
    - `coarse_type`
    - `corrected_em`
    - `corrected_f1`
    - `exact_match`
    - `token_f1`

### Graph: `Влияние blur sigma на скорректированный EM (seen)` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 17; saved as `artifacts_v2_controlled/analysis/figures/extraction_blur_sigma_curve.png`
- X axis: `Sigma blur`
- Y axis: `Скорректированный EM`
- Data series:
  - `Florence-2-base`
  - `Qwen2-VL-2B`
  - `Qwen2-VL-7B`
- Key values:
  - Data coincide with the blur table already listed above for `v2`
  - Explicit values available from outputs:
    - Florence-2-base:
      - `sigma 20 -> 0.08075`
      - `sigma 50 -> 0.08075`
    - Qwen2-VL-2B:
      - `sigma 20 -> 0.48075`
      - `sigma 50 -> 0.49075`
    - Qwen2-VL-7B:
      - `sigma 20 -> 0.58575`
      - `sigma 50 -> 0.57575`

### Graph: `Сравнение режимов: image-only vs image+OCR` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 18; saved as `artifacts_v2_controlled/analysis/figures/extraction_mode_compare.png`
- X axis: `Модель`
- Y axis: `Средний скорректированный EM`
- Data series:
  - `image_only`
  - `image+OCR`
- Key values:
  - Data coincide with the mode comparison table already listed above for `v2`
  - Explicit values available from outputs:
    - Qwen2-VL-2B `image_only = 0.56325`
    - Qwen2-VL-2B `image+OCR = 0.16575`
    - Qwen2-VL-7B `image_only = 0.65950`

### Graph: `Кривая extraction по эпохам (img_black, seen)` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 19; intended save path `artifacts_v2_controlled/analysis/figures/extraction_epoch_curve.png`
- X axis: `Эпоха`
- Y axis: `Скорректированный EM`
- Data series: one line per model label
- Key values:
  - [data not extracted, notebook output indicates the epoch-curve file was not available]

### Graph: `Связь MIA и extraction` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 20; saved as `artifacts_v2_controlled/analysis/figures/mia_vs_extraction_scatter.png`
- X axis: `MIA AUC-ROC по loss`
- Y axis: `Extraction скорректированный EM (img_black, seen)`
- Data series: one point per model
- Key values:
  - Data coincide with the scatter table already listed above for `v2`
  - Florence-2-base:
    - `mia_auc = 0.988675`
    - `extraction_corrected_em = 0.08075`
  - Qwen2-VL-2B:
    - `mia_auc = 0.920700`
    - `extraction_corrected_em = 0.47575`
  - Qwen2-VL-7B:
    - `mia_auc = 0.941725`
    - `extraction_corrected_em = 0.57575`

### Graph: `qualitative_successful_extractions` (`24_final_analysis_v2.ipynb`)
- File/cell: cell 23; saved as `artifacts_v2_controlled/analysis/figures/qualitative_successful_extractions.png`
- X axis: none
- Y axis: none
- Data series: image grid of successful extraction examples
- Key values:
  - [this is a qualitative figure; no numeric axes]
