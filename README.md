# EFS Quality Control and Daily Feed-Intake Correction

Python implementation for processing longitudinal electronic feeding-station (EFS) records. The pipeline reconstructs daily body-weight trajectories, identifies anomalous feeding visits, adjusts daily feed intake with a mixed-effects model, and imputes missing daily feed-intake values.

## Requirements

Python 3.10+ recommended. Install dependencies directly (no separate requirements file is needed):

```bash
python -m pip install "numpy>=1.24,<3" "pandas>=2.0,<3" "scipy>=1.10,<2" "scikit-learn>=1.2,<2" "statsmodels>=0.14,<1" "matplotlib>=3.7,<4"
```

These are broad compatibility ranges, **not** a record of the original analysis environment. For formal reproducibility, archive and report the exact versions used to generate results.

## Input

Input must be a visit-level CSV. Required columns:

| Column | Meaning |
| --- | --- |
| `lifenumber` | Animal ID (its first two characters are used as the breed code) |
| `location` | Testing location / station |
| `visit_time` | Visit date or datetime |
| `duration` | Visit duration in seconds |
| `weight` | Body weight (typically kg) |
| `feed_intake` | Intake per visit (typically g) |

If `start_time` and `end_time` are provided, the program calculates duration and visit date from the timestamps. If `start_feed` and `end_feed` are provided, it calculates visit intake as `start_feed - end_feed`. Confirm data direction, unit conventions, and column mappings before use. Each file may contain multiple breeds; the `--farm` argument is an output label, not a filter.

## Run

```bash
python efs_qc.py --feed visits.csv --farm EXAMPLE --output_dir ./results
```

To generate only the body-weight and population-reference diagnostics:

```bash
python efs_qc.py --feed visits.csv --farm EXAMPLE --output_dir ./references --reference-only
```

## Processing steps

1. **Body weight:** Per-animal quadratic robust regression with Tukey biweights; retain individual weight observations with weight > 0.25 and dates with a maximum retained biweight >= 0.90. Require daily growth-fit R² >= 0.99. Fill internal days with linear interpolation and handle edge days using bounded natural cubic splines.
2. **Feeding references:** Per-animal centered 7-day rolling medians for feed intake, duration, and feeding rate. Estimate breed-specific, body-weight-conditioned references using 1-kg bins supported by at least five distinct animals; smooth reference curves.
3. **Anomaly detection:** Separate Isolation Forests for intake and duration, followed by a feeding-rate model on eligible visits. The source explicitly sets `contamination=0.02` and `random_state=42`; number of trees is the installed scikit-learn default. Labels use a lower-tail score threshold based on a one-sided median absolute deviation multiplier of 4.5.
4. **Daily features:** Classify detected anomalies as low, high, or unknown relative to within-animal references; summarize direction fractions and severity scores at animal-day level.
5. **Feed-intake adjustment:** Fit a random-intercept mixed-effects model with location, body weight, ADG, and anomaly-related predictors. Subtract anomaly-related fixed-effect terms from anomaly-free feed intake and clip at zero.
6. **Missing days:** Use LOWESS when >=4 observed days are present; otherwise use linear interpolation where >=2 valid days are available.

## Outputs

- `weight_ADFI.csv` — daily `id`, `location`, `time`, reconstructed `weight`, corrected `ADFI`, and `breed`.
- `<farm>_weight_ADFI_<date>.csv` — dated copy of the final daily output.
- `model_fit_audit.csv` — model fitting status and retained predictors.
- `daily_stats_after_mixed_model.csv` — animal-day covariates and adjustments before filling missing days.
- `reference_bin_stats.csv`, `reference_coverage_summary.csv` — population reference diagnostics.
- `audit/` and other CSV files — record-level QC, anomaly labels, exclusions, and summary statistics.

All outputs are created under `--output_dir`.

