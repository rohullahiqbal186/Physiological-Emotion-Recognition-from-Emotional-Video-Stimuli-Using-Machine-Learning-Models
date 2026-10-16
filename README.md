from pathlib import Path

readme = r"""# Physiological Emotion Recognition from Emotional Video Stimuli

A reproducible machine-learning benchmark for classifying **NEUTRAL**, **POSITIVE/RELAXING**, and **ANXIETY** stimulus conditions from ECG and electrodermal activity (EDA) signals.

This repository accompanies the conference manuscript:

> **Physiological Emotion Recognition from Emotional Video Stimuli Using Machine Learning Models**  
> Rohullah Iqbal, Awais Khan, and Yunyoung Nam  
> Department of Computer Science and Engineering / ICT Convergence, Soonchunhyang University, South Korea.

The main implementation is the Jupyter notebook `FINAL_Normalization_and_without.ipynb`.

## Dataset

- **Dataset:** [Physiological Responses to Emotional Video Stimuli: ECG, EDA, and Temperature Data](https://data.mendeley.com/datasets/8dr667fxhz/1)
- **DOI:** [10.17632/8dr667fxhz.1](https://doi.org/10.17632/8dr667fxhz.1)
- **License:** CC BY 4.0, as listed by the Mendeley Data record.
- **Dataset paper:** Pataca, A. O., Al-Jumaili, S., Coelho, P. J., Garcia, N. M., Albuquerque, C., & Pires, I. M. (2026). *Physiological responses to emotional video stimuli: ECG, EDA, and temperature data*. Data in Brief, 66, 112720. https://doi.org/10.1016/j.dib.2026.112720

The dataset contains synchronized ECG, EDA, skin-temperature, and accelerometer recordings from 44 adult participants across 18 video stimuli. Each participant folder contains files such as `bitalino_1.txt` through `bitalino_18.txt`, as well as participant metadata.

**Do not commit the raw dataset to this repository.** Download it from Mendeley Data and configure the local dataset path in the notebook. Follow the dataset license and citation requirements.

## Task defined in this study

The notebook uses three selected video conditions:

| Video ID | Study label | Numeric label |
|---:|---|---:|
| 1 | NEUTRAL | 0 |
| 3 | POSITIVE_RELAXING | 1 |
| 5 | ANXIETY | 2 |

Participants are retained only when all three selected recordings are successfully processed. The manuscript reports 43 complete participants after excluding participant 27 because the selected recordings could not be parsed into numeric samples. This yields **129 trials** (43 participants × 3 selected trials), with 43 trials per class.

**Important interpretation:** the labels represent the intended emotional condition of each selected video. They are not participant self-reports. Because one selected video represents each class, results should be described as classification of these three stimulus conditions, not proof of generalization to all neutral, relaxing, or anxiety-inducing content.

## Methodology implemented in the notebook

1. **Load raw BITalino recordings.** Parse timestamped text files and extract ECG, EDA, temperature, and acceleration channels.
2. **Explore signals.** Plot raw physiological channels for an example participant and summarize signal statistics.
3. **Preprocess each trial.** Trim 2 seconds from both ends, interpolate missing values, and apply channel-specific filtering.
4. **Extract features.** Compute statistical and temporal features over the full trimmed trial and, when long enough, over its early, middle, and late thirds. ECG features include signal statistics and heart-rate/HRV descriptors; EDA features include tonic/phasic and skin-conductance-response (SCR) descriptors. The notebook initially creates a 456-feature all-signal table.
5. **Select ECG + EDA.** Keep ECG- and EDA-related features for the primary experiment, excluding temperature- and accelerometer-derived features.
6. **Compare normalization choices.** The notebook supports no subject normalization, subject-wise centering, and subject-wise z-score normalization. The configured primary setting is `zscore`.
7. **Train and evaluate models.** Use median imputation fitted inside each fold and `StratifiedGroupKFold` with participant identity as the group. The configured evaluation is 5 folds × 5 repeats.
8. **Report results.** Compare accuracy, balanced accuracy, macro-F1, pooled confusion matrices, and class-wise performance.

### Models

The notebook attempts to benchmark:
- CatBoost
- LightGBM
- XGBoost
- Histogram-based Gradient Boosting
- TabPFN (optional; availability depends on the installed package and its setup)

The saved notebook output shown in the submitted file contains completed results for CatBoost, LightGBM, XGBoost, and Histogram-based Gradient Boosting. TabPFN is not present in that recorded run.

## Results recorded in the submitted notebook

The following values are read from the notebook's saved execution output. They are not a guarantee that a fresh run on another environment will reproduce the exact values.

| Model | Accuracy (%) | Balanced accuracy (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| XGBoost | 88.44 ± 7.65 | 88.44 ± 7.65 | 88.50 ± 7.51 |
| CatBoost | 88.19 ± 5.38 | 88.19 ± 5.38 | 88.19 ± 5.35 |
| LightGBM | 87.50 ± 6.37 | 87.50 ± 6.37 | 87.49 ± 6.40 |
| Histogram-based Gradient Boosting | 87.20 ± 6.18 | 87.20 ± 6.18 | 87.15 ± 6.22 |

Values are mean ± standard deviation across the 25 fold evaluations. The stored pooled confusion matrix for XGBoost contains 645 predictions across five repetitions:

| True class | Predicted neutral | Predicted positive/relaxing | Predicted anxiety |
|---|---:|---:|---:|
| NEUTRAL | 181 | 1 | 33 |
| POSITIVE_RELAXING | 5 | 208 | 2 |
| ANXIETY | 33 | 0 | 182 |

The largest confusion is between NEUTRAL and ANXIETY. POSITIVE_RELAXING is the most consistently classified class in this saved run.

## Evaluation and normalization caveat

The notebook uses participant-wise normalization calculated from each participant's selected recordings **before** cross-validation. This does not use class labels, and the participant groups remain disjoint between train and test folds. However, normalization statistics for a held-out participant are computed using that participant's available recordings, including the selected trials that are later evaluated. This is a **subject-disjoint, transductive normalization** protocol, not a strict inductive protocol for a completely unseen participant with no participant-specific calibration data.

This distinction should be reported clearly. For a stronger deployment-oriented evaluation, add a strict inductive baseline that does not use the held-out participant's other trials to estimate normalization statistics. Also consider leave-one-subject-out evaluation and validation on additional videos per emotion category.

## Requirements

The notebook uses Python with the following core packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter
```

Optional model packages:

```bash
pip install catboost lightgbm xgboost
```

TabPFN installation and use can depend on its current package requirements and authentication/licensing workflow. Install and configure it according to the official TabPFN documentation if you want to include it.

## Run the notebook

1. Download and extract the dataset from [Mendeley Data](https://data.mendeley.com/datasets/8dr667fxhz/1).
2. Place the extracted participant folders in a local data directory.
3. Open `FINAL_Normalization_and_without.ipynb` in Jupyter or Google Colab.
4. Update `DATASET_ROOT` in the configuration cell to the directory containing the participant folders.
5. Run the notebook from top to bottom.
6. Check the reported participant count, missing/failed recordings, feature counts, available models, and saved evaluation output before interpreting results.

Example Windows path:

```python
DATASET_ROOT = Path(r"D:\path\to\Physiological Responses dataset")
```

The notebook's current path is specific to the original development machine and must be changed on another computer.

## Suggested repository structure

```text
physiological-emotion-recognition/
├── README.md
├── FINAL_Normalization_and_without.ipynb
├── Rohullah_Iqbal_Conference_Manuscript.pdf
├── requirements.txt
├── .gitignore
└── data/
    └── README.md   # instructions/link only; do not commit raw recordings
```

## Reproducibility notes

- Set and report the random seed, fold count, repeat count, normalization mode, and package versions.
- Keep all recordings from a participant in the same cross-validation partition.
- Fit imputation and any learned preprocessing only on the training fold.
- Report the exact number of participants/trials retained and any parsing failures.
- Preserve the distinction between stimulus-intended labels and self-reported emotions.
- Avoid presenting results from the same selected videos as evidence of broad emotion generalization.

## Citation

If you use the dataset, cite the dataset record and its associated data article:

> Pataca, A. O., Al-Jumaili, S., Coelho, P. J., Garcia, N. M., Albuquerque, C., & Pires, I. M. (2026). *Physiological responses to emotional video stimuli: ECG, EDA, and temperature data*. Data in Brief, 66, 112720. https://doi.org/10.1016/j.dib.2026.112720

Please also cite the dataset DOI: https://doi.org/10.17632/8dr667fxhz.1

## Code license

No separate license for the code is specified here. Add a `LICENSE` file before inviting reuse, and choose a license appropriate for your intended sharing terms. The dataset's CC BY 4.0 license does not automatically determine the license for your code or manuscript.
"""
out = Path("/mnt/data/README_emotion_recognition.md")
out.write_text(readme, encoding="utf-8")
print(f"Created {out} ({len(readme.splitlines())} lines)")
