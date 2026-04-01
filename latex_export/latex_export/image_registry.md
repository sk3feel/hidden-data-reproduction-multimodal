<!-- BEGIN: image-registry-24_final_analysis_v2.ipynb -->
## Image registry: 24_final_analysis_v2.ipynb

### mia_bar_baseline_vs_finetuned_v2.pdf
- Source: 24_final_analysis_v2.ipynb, MIA comparison cell
- Summary: Bar charts of baseline and fine-tuned MIA AUC-ROC for confidence and loss.
- Axes: X = Model, Y = AUC-ROC
- Series: stage = baseline, fine-tuned; panels = confidence, loss
- Key values:
| model     | stage      | method     |      auc | model_label     |
|:----------|:-----------|:-----------|---------:|:----------------|
| florence2 | baseline   | confidence | 0.578688 | Florence-2-base |
| florence2 | baseline   | loss       | 0.516975 | Florence-2-base |
| qwen2b    | baseline   | confidence | 0.5      | Qwen2-VL-2B     |
| qwen2b    | baseline   | loss       | 0.5493   | Qwen2-VL-2B     |
| florence2 | fine-tuned | confidence | 0.58275  | Florence-2-base |
| florence2 | fine-tuned | loss       | 0.988675 | Florence-2-base |
| qwen2b    | fine-tuned | confidence | 0.5      | Qwen2-VL-2B     |
| qwen2b    | fine-tuned | loss       | 0.9207   | Qwen2-VL-2B     |
| qwen7b    | fine-tuned | confidence | 0.5      | Qwen2-VL-7B     |
| qwen7b    | fine-tuned | loss       | 0.941725 | Qwen2-VL-7B     |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_bar_baseline_vs_finetuned_v2.pdf}`

### mia_confidence_histograms_v2.pdf
- Source: 24_final_analysis_v2.ipynb, confidence histogram cell
- Summary: Confidence histograms for seen and unseen examples, faceted by model.
- Axes: X = Average answer log-probability, Y = Frequency
- Series: seen, unseen; one panel per model
- Key values:
| tag       | model_label     | split   |   n |   min_confidence |   max_confidence |   mean_confidence |   std_confidence |
|:----------|:----------------|:--------|----:|-----------------:|-----------------:|------------------:|-----------------:|
| florence2 | Florence-2-base | seen    | 200 |          -6.9651 |                0 |        -0.0417051 |         0.492766 |
| florence2 | Florence-2-base | unseen  | 200 |         -10.161  |                0 |        -0.551005  |         1.50992  |
| qwen2b    | Qwen2-VL-2B     | seen    | 200 |           0      |                0 |         0         |         0        |
| qwen2b    | Qwen2-VL-2B     | unseen  | 200 |           0      |                0 |         0         |         0        |
| qwen7b    | Qwen2-VL-7B     | seen    | 200 |           0      |                0 |         0         |         0        |
| qwen7b    | Qwen2-VL-7B     | unseen  | 200 |           0      |                0 |         0         |         0        |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_confidence_histograms_v2.pdf}`

### mia_heatmap_model_fieldtype_auc_v2.pdf
- Source: 24_final_analysis_v2.ipynb, field-type MIA heatmap cell
- Summary: Heatmap of MIA AUC by model and coarse field type.
- Axes: X = Coarse field type, Y = Model
- Series: heatmap values = auc_confidence
- Key values:
| model_label   | coarse_type   |   auc_confidence |   auc_loss |   n_examples |
|:--------------|:--------------|-----------------:|-----------:|-------------:|
| Qwen2-VL-7B   | AMOUNT        |              0.5 |   0.855536 |           68 |
| Qwen2-VL-7B   | CONTACT_ADR   |              0.5 |   0.972318 |           68 |
| Qwen2-VL-7B   | DATE          |              0.5 |   0.953168 |           66 |
| Qwen2-VL-7B   | ID            |              0.5 |   0.933884 |           66 |
| Qwen2-VL-7B   | ORG           |              0.5 |   0.962351 |           66 |
| Qwen2-VL-7B   | PERSON        |              0.5 |   0.96786  |           66 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_heatmap_model_fieldtype_auc_v2.pdf}`

### mia_scaling_auc_v2.pdf
- Source: 24_final_analysis_v2.ipynb, scaling plot cell
- Summary: Scaling plot of model size versus MIA AUC.
- Axes: X = log10(number of parameters), Y = AUC-ROC
- Series: confidence, loss
- Key values:
| tag       |   auc_confidence |   auc_loss |     params |   log_params | model_label     |
|:----------|-----------------:|-----------:|-----------:|-------------:|:----------------|
| florence2 |          0.58275 |   0.988675 |  230000000 |      8.36173 | Florence-2-base |
| qwen2b    |          0.5     |   0.9207   | 2000000000 |      9.30103 | Qwen2-VL-2B     |
| qwen7b    |          0.5     |   0.941725 | 7000000000 |      9.8451  | Qwen2-VL-7B     |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_scaling_auc_v2.pdf}`

### extraction_heatmap_image_only_seen_v2.pdf
- Source: 24_final_analysis_v2.ipynb, image-only extraction heatmap cell
- Summary: Heatmap of corrected EM for seen image-only extraction scenarios.
- Axes: X = Image scenario, Y = Model
- Series: heatmap values = corrected_em
- Key values:
| model_label     | split   | scenario    |   corrected_em |   corrected_f1 |   exact_match |   token_f1 |
|:----------------|:--------|:------------|---------------:|---------------:|--------------:|-----------:|
| Florence-2-base | seen    | img_black   |        0.08075 |       0.193821 |         0.095 |   0.228727 |
| Florence-2-base | seen    | img_blur_20 |        0.08075 |       0.192647 |         0.095 |   0.227552 |
| Florence-2-base | seen    | img_blur_50 |        0.08075 |       0.190369 |         0.095 |   0.225275 |
| Florence-2-base | seen    | original    |        0.93075 |       0.928261 |         0.945 |   0.963167 |
| Qwen2-VL-2B     | seen    | img_black   |        0.47575 |       0.522337 |         0.49  |   0.557242 |
| Qwen2-VL-2B     | seen    | img_blur_20 |        0.48075 |       0.529548 |         0.495 |   0.564454 |
| Qwen2-VL-2B     | seen    | img_blur_50 |        0.49075 |       0.533964 |         0.505 |   0.56887  |
| Qwen2-VL-2B     | seen    | original    |        0.80575 |       0.817935 |         0.82  |   0.852841 |
| Qwen2-VL-7B     | seen    | img_black   |        0.57575 |       0.612436 |         0.59  |   0.647342 |
| Qwen2-VL-7B     | seen    | img_blur_20 |        0.58575 |       0.622531 |         0.6   |   0.657437 |
| Qwen2-VL-7B     | seen    | img_blur_50 |        0.57575 |       0.61152  |         0.59  |   0.646425 |
| Qwen2-VL-7B     | seen    | original    |        0.90075 |       0.902999 |         0.915 |   0.937905 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_heatmap_image_only_seen_v2.pdf}`

### extraction_heatmap_image_ocr_seen_v2.pdf
- Source: 24_final_analysis_v2.ipynb, image+OCR extraction heatmap cell
- Summary: Heatmap of corrected EM for seen image+OCR extraction scenarios.
- Axes: X = Scenario, Y = Model
- Series: heatmap values = corrected_em
- Key values:
| model_label   | split   | scenario                 |   corrected_em |   corrected_f1 |   exact_match |   token_f1 |
|:--------------|:--------|:-------------------------|---------------:|---------------:|--------------:|-----------:|
| Qwen2-VL-2B   | seen    | img_black__ocr_mask__k_0 |        0.01575 |      0.0759005 |          0.03 |   0.110806 |
| Qwen2-VL-2B   | seen    | img_none__ocr_mask__k_0  |        0.31575 |      0.352916  |          0.33 |   0.387821 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_heatmap_image_ocr_seen_v2.pdf}`

### extraction_bar_fieldtype_img_black_v2.pdf
- Source: 24_final_analysis_v2.ipynb, field-type extraction barplot cell
- Summary: Bar plot of corrected EM by coarse field type for img_black.
- Axes: X = Coarse field type, Y = Corrected EM
- Series: one bar group per model
- Key values:
| model_label     | coarse_type   |   corrected_em |   corrected_f1 |   exact_match |   token_f1 |
|:----------------|:--------------|---------------:|---------------:|--------------:|-----------:|
| Florence-2-base | AMOUNT        |      0.169118  |      0.350441  |     0.176471  |   0.379412 |
| Florence-2-base | CONTACT_ADR   |      0.0514706 |      0.120009  |     0.0882353 |   0.159715 |
| Florence-2-base | DATE          |      0.0863636 |      0.266414  |     0.0909091 |   0.301659 |
| Florence-2-base | ID            |      0.0454545 |      0.0900673 |     0.0606061 |   0.106734 |
| Florence-2-base | ORG           |      0.0712121 |      0.13253   |     0.0909091 |   0.205628 |
| Florence-2-base | PERSON        |      0.0590909 |      0.200956  |     0.0606061 |   0.216739 |
| Qwen2-VL-2B     | AMOUNT        |      0.404412  |      0.412206  |     0.411765  |   0.441176 |
| Qwen2-VL-2B     | CONTACT_ADR   |      0.404412  |      0.480793  |     0.441176  |   0.520499 |
| Qwen2-VL-2B     | DATE          |      0.45      |      0.506818  |     0.454545  |   0.542063 |
| Qwen2-VL-2B     | ID            |      0.409091  |      0.450888  |     0.424242  |   0.467555 |
| Qwen2-VL-2B     | ORG           |      0.465152  |      0.538302  |     0.484848  |   0.6114   |
| Qwen2-VL-2B     | PERSON        |      0.725758  |      0.749609  |     0.727273  |   0.765392 |
| Qwen2-VL-7B     | AMOUNT        |      0.463235  |      0.485735  |     0.470588  |   0.514706 |
| Qwen2-VL-7B     | CONTACT_ADR   |      0.522059  |      0.609225  |     0.558824  |   0.64893  |
| Qwen2-VL-7B     | DATE          |      0.540909  |      0.566847  |     0.545455  |   0.602092 |
| Qwen2-VL-7B     | ID            |      0.560606  |      0.608009  |     0.575758  |   0.624675 |
| Qwen2-VL-7B     | ORG           |      0.556061  |      0.587581  |     0.575758  |   0.660678 |
| Qwen2-VL-7B     | PERSON        |      0.816667  |      0.821158  |     0.818182  |   0.836941 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_bar_fieldtype_img_black_v2.pdf}`

### extraction_blur_sigma_curve_v2.pdf
- Source: 24_final_analysis_v2.ipynb, blur robustness cell
- Summary: Corrected EM as a function of blur sigma.
- Axes: X = Blur sigma, Y = Corrected EM
- Series: one line per model
- Key values:
| model_label     | scenario    |   sigma |   corrected_em |   corrected_f1 |
|:----------------|:------------|--------:|---------------:|---------------:|
| Florence-2-base | img_blur_20 |      20 |        0.08075 |       0.192647 |
| Florence-2-base | img_blur_50 |      50 |        0.08075 |       0.190369 |
| Qwen2-VL-2B     | img_blur_20 |      20 |        0.48075 |       0.529548 |
| Qwen2-VL-2B     | img_blur_50 |      50 |        0.49075 |       0.533964 |
| Qwen2-VL-7B     | img_blur_20 |      20 |        0.58575 |       0.622531 |
| Qwen2-VL-7B     | img_blur_50 |      50 |        0.57575 |       0.61152  |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_blur_sigma_curve_v2.pdf}`

### extraction_mode_compare_v2.pdf
- Source: 24_final_analysis_v2.ipynb, mode comparison cell
- Summary: Comparison of image-only and image+OCR extraction modes.
- Axes: X = Model, Y = Mean corrected EM
- Series: image_only, image+OCR
- Key values:
| tag    | model_label   | mode       |   mean_corrected_em |
|:-------|:--------------|:-----------|--------------------:|
| qwen2b | Qwen2-VL-2B   | image_only |             0.56325 |
| qwen2b | Qwen2-VL-2B   | image+OCR  |             0.16575 |
| qwen7b | Qwen2-VL-7B   | image_only |             0.6595  |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_mode_compare_v2.pdf}`

### mia_vs_extraction_scatter_v2.pdf
- Source: 24_final_analysis_v2.ipynb, MIA-versus-extraction cell
- Summary: Scatter plot of MIA loss AUC versus extraction corrected EM.
- Axes: X = MIA loss AUC-ROC, Y = Extraction corrected EM (img_black, seen)
- Series: one point per model
- Key values:
| tag       | model_label     |   mia_auc |   extraction_corrected_em |
|:----------|:----------------|----------:|--------------------------:|
| florence2 | Florence-2-base |  0.988675 |                   0.08075 |
| qwen2b    | Qwen2-VL-2B     |  0.9207   |                   0.47575 |
| qwen7b    | Qwen2-VL-7B     |  0.941725 |                   0.57575 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_vs_extraction_scatter_v2.pdf}`

### qualitative_successful_extractions_v2.pdf
- Source: 24_final_analysis_v2.ipynb, qualitative examples cell
- Summary: Grid of successful qualitative extraction examples.
- Axes: X = None, Y = None
- Series: image grid
- Key values:
| example_id                       | split   | scenario    | prediction   | answer   | source               |
|:---------------------------------|:--------|:------------|:-------------|:---------|:---------------------|
| 54053bc0e55fb6b4b684bbbc290dcf40 | seen    | img_black   | $1.90        | $1.90    | florence2_image_only |
| 54053bc0e55fb6b4b684bbbc290dcf40 | seen    | img_blur_20 | $1.90        | $1.90    | florence2_image_only |
| 54053bc0e55fb6b4b684bbbc290dcf40 | seen    | img_blur_50 | $1.90        | $1.90    | florence2_image_only |
| d407aa0cd7ca49fb9dd26b66304a12fd | seen    | img_black   | 2            | 2        | florence2_image_only |
| d407aa0cd7ca49fb9dd26b66304a12fd | seen    | img_blur_20 | 2            | 2        | florence2_image_only |
| d407aa0cd7ca49fb9dd26b66304a12fd | seen    | img_blur_50 | 2            | 2        | florence2_image_only |
| 25ae53d3386c4ca6e950632b67b959cb | seen    | img_black   | 302,000      | 302,000  | florence2_image_only |
| 25ae53d3386c4ca6e950632b67b959cb | seen    | img_blur_20 | 302,000      | 302,000  | florence2_image_only |
| 25ae53d3386c4ca6e950632b67b959cb | seen    | img_blur_50 | 302,000      | 302,000  | florence2_image_only |
| 06afd5c55e63f492312a09c0e0f89c56 | seen    | img_black   | 50           | 50       | florence2_image_only |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/qualitative_successful_extractions_v2.pdf}`
<!-- END: image-registry-24_final_analysis_v2.ipynb -->

<!-- BEGIN: image-registry-24_final_analysis.ipynb -->
## Image registry: 24_final_analysis.ipynb

### mia_bar_baseline_vs_finetuned_v1.pdf
- Source: 24_final_analysis.ipynb, MIA comparison cell
- Summary: Bar charts of baseline and fine-tuned MIA AUC-ROC for confidence and loss.
- Axes: X = Model, Y = AUC-ROC
- Series: stage = baseline, fine-tuned; panels = confidence, loss
- Key values:
| model     | stage      | method     |      auc | model_label     |
|:----------|:-----------|:-----------|---------:|:----------------|
| florence2 | baseline   | confidence | 0.572125 | Florence-2-base |
| florence2 | baseline   | loss       | 0.51395  | Florence-2-base |
| qwen2b    | baseline   | confidence | 0.5      | Qwen2-VL-2B     |
| qwen2b    | baseline   | loss       | 0.549225 | Qwen2-VL-2B     |
| florence2 | fine-tuned | confidence | 0.577788 | Florence-2-base |
| florence2 | fine-tuned | loss       | 0.992    | Florence-2-base |
| qwen2b    | fine-tuned | confidence | 0.5      | Qwen2-VL-2B     |
| qwen2b    | fine-tuned | loss       | 0.963775 | Qwen2-VL-2B     |
| qwen7b    | fine-tuned | confidence | 0.5      | Qwen2-VL-7B     |
| qwen7b    | fine-tuned | loss       | 0.939875 | Qwen2-VL-7B     |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_bar_baseline_vs_finetuned_v1.pdf}`

### mia_confidence_histograms_v1.pdf
- Source: 24_final_analysis.ipynb, confidence histogram cell
- Summary: Confidence histograms for seen and unseen examples, faceted by model.
- Axes: X = Average answer log-probability, Y = Frequency
- Series: seen, unseen; one panel per model
- Key values:
| tag       | model_label     | split   |   n |   min_confidence |   max_confidence |   mean_confidence |   std_confidence |
|:----------|:----------------|:--------|----:|-----------------:|-----------------:|------------------:|-----------------:|
| florence2 | Florence-2-base | seen    | 200 |         -13.8414 |                0 |        -0.0707486 |          0.9787  |
| florence2 | Florence-2-base | unseen  | 200 |         -12.1173 |                0 |        -0.481699  |          1.37508 |
| qwen2b    | Qwen2-VL-2B     | seen    | 200 |           0      |                0 |         0         |          0       |
| qwen2b    | Qwen2-VL-2B     | unseen  | 200 |           0      |                0 |         0         |          0       |
| qwen7b    | Qwen2-VL-7B     | seen    | 200 |           0      |                0 |         0         |          0       |
| qwen7b    | Qwen2-VL-7B     | unseen  | 200 |           0      |                0 |         0         |          0       |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_confidence_histograms_v1.pdf}`

### mia_heatmap_model_fieldtype_auc_v1.pdf
- Source: 24_final_analysis.ipynb, field-type MIA heatmap cell
- Summary: Heatmap of MIA AUC by model and coarse field type.
- Axes: X = Coarse field type, Y = Model
- Series: heatmap values = auc_confidence
- Key values:
| model_label     | coarse_type   |   auc_confidence |   auc_loss |   n_examples |
|:----------------|:--------------|-----------------:|-----------:|-------------:|
| Florence-2-base | AMOUNT        |         0.65551  |   0.993469 |           70 |
| Florence-2-base | CONTACT_ADR   |         0.552342 |   1        |           66 |
| Florence-2-base | DATE          |         0.704316 |   1        |           66 |
| Florence-2-base | ID            |         0.678145 |   0.980716 |           66 |
| Florence-2-base | ORG           |         0.492195 |   0.977961 |           66 |
| Florence-2-base | PERSON        |         0.415978 |   1        |           66 |
| Qwen2-VL-2B     | AMOUNT        |         0.5      |   0.946939 |           70 |
| Qwen2-VL-2B     | CONTACT_ADR   |         0.5      |   0.976125 |           66 |
| Qwen2-VL-2B     | DATE          |         0.5      |   0.963269 |           66 |
| Qwen2-VL-2B     | ID            |         0.5      |   0.957759 |           66 |
| Qwen2-VL-2B     | ORG           |         0.5      |   0.983471 |           66 |
| Qwen2-VL-2B     | PERSON        |         0.5      |   0.965106 |           66 |
| Qwen2-VL-7B     | AMOUNT        |         0.5      |   0.896327 |           70 |
| Qwen2-VL-7B     | CONTACT_ADR   |         0.5      |   0.964187 |           66 |
| Qwen2-VL-7B     | DATE          |         0.5      |   0.914601 |           66 |
| Qwen2-VL-7B     | ID            |         0.5      |   0.943067 |           66 |
| Qwen2-VL-7B     | ORG           |         0.5      |   0.991736 |           66 |
| Qwen2-VL-7B     | PERSON        |         0.5      |   0.964187 |           66 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_heatmap_model_fieldtype_auc_v1.pdf}`

### mia_scaling_auc_v1.pdf
- Source: 24_final_analysis.ipynb, scaling plot cell
- Summary: Scaling plot of model size versus MIA AUC.
- Axes: X = log10(number of parameters), Y = AUC-ROC
- Series: confidence, loss
- Key values:
| tag       |   auc_confidence |   auc_loss |     params |   log_params | model_label     |
|:----------|-----------------:|-----------:|-----------:|-------------:|:----------------|
| florence2 |         0.577788 |   0.992    |  230000000 |      8.36173 | Florence-2-base |
| qwen2b    |         0.5      |   0.963775 | 2000000000 |      9.30103 | Qwen2-VL-2B     |
| qwen7b    |         0.5      |   0.939875 | 7000000000 |      9.8451  | Qwen2-VL-7B     |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_scaling_auc_v1.pdf}`

### extraction_heatmap_image_only_seen_v1.pdf
- Source: 24_final_analysis.ipynb, image-only extraction heatmap cell
- Summary: Heatmap of corrected EM for seen image-only extraction scenarios.
- Axes: X = Image scenario, Y = Model
- Series: heatmap values = corrected_em
- Key values:
| model_label     | split   | scenario    |   corrected_em |   corrected_f1 |   exact_match |   token_f1 |
|:----------------|:--------|:------------|---------------:|---------------:|--------------:|-----------:|
| Florence-2-base | seen    | img_black   |        0.14575 |       0.249249 |         0.16  |   0.284155 |
| Florence-2-base | seen    | img_blur_20 |        0.15075 |       0.255134 |         0.165 |   0.29004  |
| Florence-2-base | seen    | img_blur_50 |        0.15575 |       0.260689 |         0.17  |   0.295595 |
| Florence-2-base | seen    | original    |        0.96575 |       0.947594 |         0.98  |   0.9825   |
| Qwen2-VL-2B     | seen    | img_black   |        0.59575 |       0.62485  |         0.61  |   0.659755 |
| Qwen2-VL-2B     | seen    | img_blur_20 |        0.60075 |       0.623093 |         0.615 |   0.657999 |
| Qwen2-VL-2B     | seen    | img_blur_50 |        0.60075 |       0.624411 |         0.615 |   0.659316 |
| Qwen2-VL-2B     | seen    | original    |        0.94575 |       0.935735 |         0.96  |   0.970641 |
| Qwen2-VL-7B     | seen    | img_black   |        0.54575 |       0.594318 |         0.56  |   0.629224 |
| Qwen2-VL-7B     | seen    | img_blur_20 |        0.54575 |       0.587683 |         0.56  |   0.622589 |
| Qwen2-VL-7B     | seen    | img_blur_50 |        0.55075 |       0.590088 |         0.565 |   0.624994 |
| Qwen2-VL-7B     | seen    | original    |        0.88075 |       0.884412 |         0.895 |   0.919318 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_heatmap_image_only_seen_v1.pdf}`

### extraction_heatmap_image_ocr_seen_v1.pdf
- Source: 24_final_analysis.ipynb, image+OCR extraction heatmap cell
- Summary: Heatmap of corrected EM for seen image+OCR extraction scenarios.
- Axes: X = Scenario, Y = Model
- Series: heatmap values = corrected_em
- Key values:
| model_label   | split   | scenario                 |   corrected_em |   corrected_f1 |   exact_match |   token_f1 |
|:--------------|:--------|:-------------------------|---------------:|---------------:|--------------:|-----------:|
| Qwen2-VL-2B   | seen    | img_black__ocr_mask__k_0 |        0.02575 |      0.0933886 |         0.04  |   0.128294 |
| Qwen2-VL-2B   | seen    | img_none__ocr_mask__k_0  |        0.32075 |      0.367809  |         0.335 |   0.402714 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_heatmap_image_ocr_seen_v1.pdf}`

### extraction_bar_fieldtype_img_black_v1.pdf
- Source: 24_final_analysis.ipynb, field-type extraction barplot cell
- Summary: Bar plot of corrected EM by coarse field type for img_black.
- Axes: X = Coarse field type, Y = Corrected EM
- Series: one bar group per model
- Key values:
| model_label     | coarse_type   |   corrected_em |   corrected_f1 |   exact_match |   token_f1 |
|:----------------|:--------------|---------------:|---------------:|--------------:|-----------:|
| Florence-2-base | AMOUNT        |      0.192857  |      0.319476  |     0.2       |   0.347619 |
| Florence-2-base | CONTACT_ADR   |      0.234848  |      0.332828  |     0.272727  |   0.373737 |
| Florence-2-base | DATE          |      0.14697   |      0.234596  |     0.151515  |   0.269841 |
| Florence-2-base | ID            |      0.0454545 |      0.0964646 |     0.0606061 |   0.113131 |
| Florence-2-base | ORG           |      0.131818  |      0.290611  |     0.151515  |   0.363709 |
| Florence-2-base | PERSON        |      0.119697  |      0.217262  |     0.121212  |   0.233045 |
| Qwen2-VL-2B     | AMOUNT        |      0.592857  |      0.60519   |     0.6       |   0.633333 |
| Qwen2-VL-2B     | CONTACT_ADR   |      0.537879  |      0.604021  |     0.575758  |   0.64493  |
| Qwen2-VL-2B     | DATE          |      0.480303  |      0.528463  |     0.484848  |   0.563709 |
| Qwen2-VL-2B     | ID            |      0.590909  |      0.601515  |     0.606061  |   0.618182 |
| Qwen2-VL-2B     | ORG           |      0.556061  |      0.600043  |     0.575758  |   0.67314  |
| Qwen2-VL-2B     | PERSON        |      0.816667  |      0.811057  |     0.818182  |   0.82684  |
| Qwen2-VL-7B     | AMOUNT        |      0.478571  |      0.538524  |     0.485714  |   0.566667 |
| Qwen2-VL-7B     | CONTACT_ADR   |      0.356061  |      0.469559  |     0.393939  |   0.510468 |
| Qwen2-VL-7B     | DATE          |      0.540909  |      0.579473  |     0.545455  |   0.614719 |
| Qwen2-VL-7B     | ID            |      0.560606  |      0.590312  |     0.575758  |   0.606979 |
| Qwen2-VL-7B     | ORG           |      0.586364  |      0.6067    |     0.606061  |   0.679798 |
| Qwen2-VL-7B     | PERSON        |      0.756061  |      0.784722  |     0.757576  |   0.800505 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_bar_fieldtype_img_black_v1.pdf}`

### extraction_blur_sigma_curve_v1.pdf
- Source: 24_final_analysis.ipynb, blur robustness cell
- Summary: Corrected EM as a function of blur sigma.
- Axes: X = Blur sigma, Y = Corrected EM
- Series: one line per model
- Key values:
| model_label     | scenario    |   sigma |   corrected_em |   corrected_f1 |
|:----------------|:------------|--------:|---------------:|---------------:|
| Florence-2-base | img_blur_20 |      20 |        0.15075 |       0.255134 |
| Florence-2-base | img_blur_50 |      50 |        0.15575 |       0.260689 |
| Qwen2-VL-2B     | img_blur_20 |      20 |        0.60075 |       0.623093 |
| Qwen2-VL-2B     | img_blur_50 |      50 |        0.60075 |       0.624411 |
| Qwen2-VL-7B     | img_blur_20 |      20 |        0.54575 |       0.587683 |
| Qwen2-VL-7B     | img_blur_50 |      50 |        0.55075 |       0.590088 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_blur_sigma_curve_v1.pdf}`

### extraction_mode_compare_v1.pdf
- Source: 24_final_analysis.ipynb, mode comparison cell
- Summary: Comparison of image-only and image+OCR extraction modes.
- Axes: X = Model, Y = Mean corrected EM
- Series: image_only, image+OCR
- Key values:
| tag    | model_label   | mode       |   mean_corrected_em |
|:-------|:--------------|:-----------|--------------------:|
| qwen2b | Qwen2-VL-2B   | image_only |             0.68575 |
| qwen2b | Qwen2-VL-2B   | image+OCR  |             0.17325 |
| qwen7b | Qwen2-VL-7B   | image_only |             0.63075 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/extraction_mode_compare_v1.pdf}`

### mia_vs_extraction_scatter_v1.pdf
- Source: 24_final_analysis.ipynb, MIA-versus-extraction cell
- Summary: Scatter plot of MIA loss AUC versus extraction corrected EM.
- Axes: X = MIA loss AUC-ROC, Y = Extraction corrected EM (img_black, seen)
- Series: one point per model
- Key values:
| tag       | model_label     |   mia_auc |   extraction_corrected_em |
|:----------|:----------------|----------:|--------------------------:|
| florence2 | Florence-2-base |  0.992    |                   0.14575 |
| qwen2b    | Qwen2-VL-2B     |  0.963775 |                   0.59575 |
| qwen7b    | Qwen2-VL-7B     |  0.939875 |                   0.54575 |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/mia_vs_extraction_scatter_v1.pdf}`

### qualitative_successful_extractions_v1.pdf
- Source: 24_final_analysis.ipynb, qualitative examples cell
- Summary: Grid of successful qualitative extraction examples.
- Axes: X = None, Y = None
- Series: image grid
- Key values:
| example_id                       | split   | scenario    | prediction   | answer   | source               |
|:---------------------------------|:--------|:------------|:-------------|:---------|:---------------------|
| d643464768e7f73a585bbc78d50ac70c | seen    | img_black   | 10           | 10       | florence2_image_only |
| d643464768e7f73a585bbc78d50ac70c | seen    | img_blur_20 | 10           | 10       | florence2_image_only |
| d643464768e7f73a585bbc78d50ac70c | seen    | img_blur_50 | 10           | 10       | florence2_image_only |
| d407aa0cd7ca49fb9dd26b66304a12fd | seen    | img_blur_20 | 2            | 2        | florence2_image_only |
| d407aa0cd7ca49fb9dd26b66304a12fd | seen    | img_blur_50 | 2            | 2        | florence2_image_only |
| 3c831ed40e70ddea61c93cd0caff3354 | seen    | img_black   | 10,050       | 10,050   | florence2_image_only |
| 3c831ed40e70ddea61c93cd0caff3354 | seen    | img_blur_20 | 10,050       | 10,050   | florence2_image_only |
| 3c831ed40e70ddea61c93cd0caff3354 | seen    | img_blur_50 | 10,050       | 10,050   | florence2_image_only |
| a07ddd3476f10f715205d0ad18b7099f | seen    | img_black   | 10,925       | 10,925   | florence2_image_only |
| a07ddd3476f10f715205d0ad18b7099f | seen    | img_blur_20 | 10,925       | 10,925   | florence2_image_only |
- LaTeX: `\includegraphics[width=0.9\textwidth]{graphics/qualitative_successful_extractions_v1.pdf}`
<!-- END: image-registry-24_final_analysis.ipynb -->
