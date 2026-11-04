# Bundled sample datasets

Four small demo datasets that the app can load. They are derived from public datasets of the UCI
Machine Learning Repository (licence CC BY 4.0, credit to the original creators) and were modified
as listed below. Full credit, DOIs and the verification notes are in `../THIRD_PARTY_NOTICES.md`.
The CSV files in this folder are not covered by this repository's MIT or CC BY licence statement for
original work; see `../LICENSE`, Section C. This README is documentation of this repository (CC BY 4.0).

| File | Rows | Features | Target = 1 means | Rows of 1 | Rows of 0 | Ratio | Source |
|---|---|---|---|---|---|---|---|
| `car_quality.csv` | 1,728 | 6 | car rated "good" | 69 | 1,659 | 24.0 to 1 | UCI Car Evaluation (Bohanec), https://doi.org/10.24432/C5JP48 |
| `credit_default.csv` | 6,000 | 23 | customer defaulted | 1,327 | 4,673 | 3.5 to 1 | UCI Default of Credit Card Clients (Yeh), https://doi.org/10.24432/C55S3H |
| `liver_disease.csv` | 583 | 10 | record is NOT a liver patient (see note) | 167 | 416 | 2.5 to 1 | UCI ILPD (Ramana and Venkateswarlu), https://doi.org/10.24432/C5D02C |
| `student_dropout.csv` | 4,424 | 36 | student is "Enrolled" (see note) | 794 | 3,630 | 4.6 to 1 | UCI Predict Students' Dropout and Academic Success (Realinho et al.), https://doi.org/10.24432/C5MC89 |

Ratio is majority rows to rare rows. The last column of every file is `target`. All values are
numeric and there are no missing cells.

## Changes made to the source data

- `car_quality.csv`: the six categorical attributes are ordinal-encoded (low to vhigh as 0 to 3,
  and the analogous orders for doors, persons, luggage boot and safety). Target is 1 for class
  "good" and 0 for all other classes. Checked row by row against the KEEL car-good file.
- `credit_default.csv`: a 6,000-row subsample of the 30,000-row source, ID column dropped. The
  default rate is the same in sample and source (0.2212). How the rows were sampled is not
  recorded, so the file cannot be regenerated from this repository. It contains the demographic
  columns SEX, EDUCATION, MARRIAGE and AGE of the original public release.
- `liver_disease.csv`: gender recoded (Female 0, Male 1), columns renamed, class relabelled.
- `student_dropout.csv`: the three-class outcome is binarised, column names tidied, and two grade
  columns rounded to four decimals.

## Two label notes you should know about

- Liver: the source marks 416 records as liver patients and 167 as non-patients. In
  `liver_disease.csv` the 416 patients are `target` 0 and the 167 non-patients are `target` 1. So
  the app treats the group without liver disease as the rare class. The card text for this sample
  used to say "spot patients with liver disease", which described the opposite class. It is fixed:
  the card now says "find the less common group: records without liver disease". This was verified
  from the file's own class counts against the published counts.
- Student: the rare class (`target` 1, 794 rows) is "Enrolled". The other 3,630 rows are "Dropout"
  (1,421) and "Graduate" (2,209) together. The card text used to say "flag students at risk of
  dropping out", which did not describe this. It is fixed: the card now says "find the less common
  group: currently enrolled students". The 4.6 to 1 ratio is Enrolled against everyone else, not
  Dropout.

Both card texts are in `backend/server.py` (`SAMPLE_META`) and are covered by
`backend/test_sample_cards.py`, which checks the rare class and row counts of all four samples
against the card text.

## What generates these files

Nothing in this repository regenerates them. `backend/gen_samples.py` writes three different,
synthetic or scikit-learn-derived example files (`transactions_fraud.csv`, `medical_screening.csv`,
`customer_churn_mixed.csv`) that are not bundled. It leaves the four files above untouched. This
was checked by running it in a throwaway copy during the review pass.

## Checksums

First 16 hex characters of the SHA-256 of each file as committed (LF line endings):

| File | SHA-256 (first 16) |
|---|---|
| `car_quality.csv` | `23495ac3523c81b3` |
| `credit_default.csv` | `8931d2d789965536` |
| `liver_disease.csv` | `c3d90c0dd5cf7366` |
| `student_dropout.csv` | `9bcdeaf39e66f9d4` |
