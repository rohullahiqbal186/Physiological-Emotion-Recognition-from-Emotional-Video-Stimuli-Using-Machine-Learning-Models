# Physiological Signal Classification: Stress Detection and Emotion Recognition

Research code for physiological-signal classification using two distinct datasets and experimental tracks:

1. **Wearable stress/rest classification** using Empatica E4 recordings from PhysioNet.
2. **Video-elicited emotion recognition** using ECG, EDA, skin temperature, and accelerometer recordings from the Mendeley Data corpus.

The repository is intended for research and reproducibility. These are separate experiments with different sensors, labels, data formats, and evaluation protocols. Their results should not be compared as if they came from the same benchmark.

## Research tracks at a glance

| Track | Dataset | Task | Main approach |
|---|---|---|---|
| A | [PhysioNet Wearable Device Dataset](https://physionet.org/content/wearable-device-dataset/1.0.1/) | Binary stress vs. rest | Handcrafted features and CNN-based feature learning |
| B | [Physiological Responses to Emotional Video Stimuli](https://data.mendeley.com/datasets/8dr667fxhz/1) | Three-class emotion recognition | ECG/EDA handcrafted features and gradient-boosted classifiers |

## Track A: Wearable stress/rest classification

### Dataset

**Dataset:** Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions*, version 1.0.1. PhysioNet. [Dataset DOI](https://doi.org/10.13026/he0v-tf17).

**Associated publication:** Hongn, A., Bosch, F., Prado, L. E., Ferrández, J. M., & Bonomini, M. P. (2025). “Wearable Physiological Signals under Acute Stress and Exercise Conditions.” *Scientific Data*. [https://doi.org/10.1038/s41597-025-04845-9](https://doi.org/10.1038/s41597-025-04845-9).

The dataset includes Empatica E4 signals such as BVP, EDA, HR, IBI, skin temperature, accelerometry, and protocol event tags. The stress/rest scripts use the `STRESS` recordings; they do not claim to classify all exercise conditions.

### Scripts

| Script | Purpose |
|---|---|
| `stress_rest_pipleine_v3.py` | Main paper-aligned handcrafted-feature stress/rest baseline |
| `stress_rest_pipleine_v2.py` | Earlier version of the handcrafted-feature baseline |
| `cnn_baseline_feature_comparison.py` | CNN feature-learning comparison, with optional handcrafted-feature comparison |
| `cnn_stress_contribution_v2.py` | CNN experiments including subject-mixed CV, leave-one-subject-out (LOSO), and protocol-version transfer |

The filename spelling `pipleine` is retained from the supplied scripts.

### Method summary

The handcrafted baseline reads Empatica CSV signals and event tags, segments the stress protocol into task/rest blocks, filters signals, extracts a 13-feature set aligned with the associated paper, applies intra-participant normalization, and evaluates classifiers using cross-validation. Random oversampling is applied within the training pipeline.

The CNN experiments learn representations from physiological windows and evaluate the resulting features using classical classifiers. Available experiments include subject-mixed evaluation, LOSO, and cross-protocol transfer. Check each script's `--help` output for the supported arguments.

The handcrafted implementation is a **paper-aligned reproduction**, not a guaranteed bit-for-bit replication, because not every low-level parameter is specified in the paper.

## Track B: Video-elicited emotion recognition

### Dataset

**Dataset:** Pataca, A. O., Albuquerque, C., & Pires, I. (2025). *Physiological Responses to Emotional Video Stimuli: ECG, EDA, and Temperature Data*, version 1. [Mendeley Data](https://data.mendeley.com/datasets/8dr667fxhz/1) · [DOI: 10.17632/8dr667fxhz.1](https://doi.org/10.17632/8dr667fxhz.1).

The dataset contains synchronized physiological recordings from 44 adult participants viewing 18 emotionally evocative video clips. Signals include ECG, EDA, skin temperature, and accelerometer data, recorded using BITalino devices. The dataset labels represent the intended elicitation conditions of the stimuli; they should not be described as participants' self-reported emotions unless supported by additional annotation.

The current three-class experiment retains participants with usable recordings for all three selected videos, yielding **43 participants and 129 trials**:

| Video ID | Class label |
|---:|---|
| 1 | `NEUTRAL` |
| 3 | `POSITIVE_RELAXING` |
| 5 | `ANXIETY` |

### Notebook

`FINAL_Normalization_and_without.ipynb`

The notebook implements an end-to-end workflow including:

1. Loading raw BITalino `.txt` files.
2. Inspecting ECG, EDA, temperature, and accelerometer signals.
3. Trimming trial edges, interpolating missing samples, filtering, and resampling selected channels.
4. Extracting ECG/HRV and EDA/SCR features, plus temperature and accelerometer descriptors.
5. Building a 456-feature representation and selecting the 324 ECG+EDA features (116 ECG and 208 EDA features) for the primary experiment.
6. Exploring feature distributions, correlations, and PCA projections.
7. Comparing no normalization, centering, or participant-wise z-score normalization, depending on the configuration.
8. Evaluating tabular classifiers using stratified group 5-fold cross-validation repeated five times, with participant identity as the grouping variable.
9. Reporting accuracy, balanced accuracy, macro-F1, confusion matrices, and class-wise results.

The notebook includes CatBoost, LightGBM, XGBoost, scikit-learn histogram-based gradient boosting, and optional TabPFN support, subject to installed packages and the notebook configuration.

### Manuscript

The repository accompanies the manuscript:

**“Physiological Emotion Recognition from Emotional Video Stimuli Using Machine Learning Models”**  
Authors listed in the manuscript: Rohullah Iqbal, Awais Khan, and Yunyoung Nam.

The manuscript reports a subject-disjoint benchmark on 43 participants and 129 trials using 324 handcrafted ECG+EDA features. Its reported best ECG+EDA result is **88.80% ± 5.90% balanced accuracy** for histogram-based gradient boosting over 25 held-out evaluations. See the manuscript itself for full experimental details and results.

**Important evaluation qualification:** participant-wise normalization estimates each participant's statistics from that participant's unlabeled recordings, including recordings that may be used for held-out prediction. Therefore, this is a subject-disjoint evaluation with participant-wise transductive normalization, not strict inductive generalization where no target-participant data are available before prediction. The emotion labels are based on the intended stimulus condition, not participant self-reports.

The manuscript is included as a research artifact/reference. Update this section with a DOI, conference, or publication status only when those details are confirmed.

## Setup

Use a separate virtual environment if the two tracks need different package versions. Python 3.10 or 3.11 is a reasonable starting point, but confirm compatibility with your installed versions of PyTorch, TabPFN, and the other packages.

### Core packages

```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib
```

For Track A, install optional model packages as needed:

```bash
pip install xgboost neurokit2
```

Install a compatible PyTorch build using the official selector: https://pytorch.org/get-started/locally/

For Track B, install the classifiers you want to evaluate:

```bash
pip install catboost lightgbm xgboost
```

TabPFN is optional and may require additional setup or license acceptance. If you do not intend to use it, leave it disabled in the notebook configuration.

## Data setup and privacy

Download the datasets from their official repositories and keep them outside version control. **Do not upload raw physiological recordings to GitHub.** Instead, link to the official dataset pages and follow their respective license and attribution requirements.

Example local project structure:

```text
repository/
├── README.md
├── requirements.txt
├── .gitignore
├── stress_rest_pipleine_v3.py
├── stress_rest_pipleine_v2.py
├── cnn_baseline_feature_comparison.py
├── cnn_stress_contribution_v2.py
├── FINAL_Normalization_and_without.ipynb
├── data/                   # local only; do not commit raw data
│   ├── Wearable_Dataset/
│   └── Physiological_Responses_dataset/
├── results/                # generated experiment outputs
└── manuscript/             # optional; only share if permitted
```

Add the data and large generated artifacts to `.gitignore`, for example:

```gitignore
data/
results/
*.zip
*.7z
*.rar
__pycache__/
.ipynb_checkpoints/
.venv/
venv/
.env
```

If you choose to include the manuscript PDF in the public repository, make sure all co-authors and any conference or publisher rules permit public sharing.

## Running the scripts

### Track A: handcrafted stress/rest baseline

Pass `--root` the directory that directly contains `STRESS/`.

```bash
python stress_rest_pipleine_v3.py --root "data/Wearable_Dataset" --output "results/paper_baseline"
```

### Track A: CNN LOSO experiment

```bash
python cnn_stress_contribution_v2.py --root "data/Wearable_Dataset" --output "results/cnn_loso" --experiment loso
```

### Track A: protocol transfer

```bash
python cnn_stress_contribution_v2.py --root "data/Wearable_Dataset" --output "results/v1_to_v2" --experiment v1_to_v2
```

Check available arguments before running:

```bash
python stress_rest_pipleine_v3.py --help
python cnn_baseline_feature_comparison.py --help
python cnn_stress_contribution_v2.py --help
```

### Track B: emotion recognition notebook

1. Download and extract the Mendeley dataset locally.
2. Open `FINAL_Normalization_and_without.ipynb`.
3. Edit `DATASET_ROOT` in the configuration cell so it points to the extracted dataset directory containing participant folders.
4. Run the notebook from top to bottom.
5. Check the printed participant count, class counts, selected feature counts, skipped models, and cross-validation results before reporting results.

The notebook's current local Windows path is only an example from the development environment; replace it with your own path.

## Evaluation and reproducibility notes

- **Keep participants grouped.** For Track B, all trials from one participant must remain in the same fold. For Track A, distinguish subject-mixed results from LOSO results.
- **Normalization protocol matters.** Participant-wise normalization can use unlabeled test-participant recordings. Report this explicitly and do not call it strict inductive unseen-subject evaluation.
- **Fit learned preprocessing on training folds.** Imputation, feature selection based on labels, resampling, and other learned transformations must be fitted only on training data.
- **Report more than accuracy.** Include balanced accuracy, macro-F1, per-class metrics, confusion matrices, and variability across folds or repetitions.
- **Avoid unsupported comparisons.** Results from different datasets, sensor systems, label definitions, or cross-validation protocols are not directly comparable.
- **Re-run experiments before publishing metrics.** This README records the results stated in the supplied manuscript; it does not guarantee that a fresh environment will reproduce them exactly.

## References

1. Hongn, A., Bosch, F., Prado, L. E., Ferrández, J. M., & Bonomini, M. P. (2025). *Wearable Physiological Signals under Acute Stress and Exercise Conditions*. Scientific Data. https://doi.org/10.1038/s41597-025-04845-9
2. Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions*, v1.0.1. PhysioNet. https://doi.org/10.13026/he0v-tf17
3. Pataca, A. O., Albuquerque, C., & Pires, I. (2025). *Physiological Responses to Emotional Video Stimuli: ECG, EDA, and Temperature Data*, version 1. Mendeley Data. https://doi.org/10.17632/8dr667fxhz.1
4. Iqbal, R., Khan, A., & Nam, Y. *Physiological Emotion Recognition from Emotional Video Stimuli Using Machine Learning Models*. Manuscript supplied with this project. Add publication metadata when confirmed.

## Code license

No code license is specified here. Add a `LICENSE` file before describing the code as open-source or granting reuse permissions. Dataset licenses are separate from the code license.
