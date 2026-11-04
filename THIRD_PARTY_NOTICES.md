# Third-party notices

This repository's own code and documentation are licensed as described in `LICENSE`. The material
below is not covered by those licences. It keeps its own terms, and credit belongs to the original
creators.

Licence statements for the datasets were read from the providers' pages on 4 November 2026 and
are not legal advice. Check the original page before relying on them.

## 1. Bundled sample datasets (`samples/*.csv`)

The four sample files are derived from datasets of the UCI Machine Learning Repository. The
repository's pages state the licence Creative Commons Attribution 4.0 International (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/) for each of them. CC BY 4.0 requires credit, a link
to the licence and an indication of changes; all three are given here. The changes are listed per
file below and again in `samples/README.md`.

| File | Source dataset | Creators | Year shown by UCI | DOI |
|---|---|---|---|---|
| `car_quality.csv` | Car Evaluation | Marko Bohanec | 1988 | https://doi.org/10.24432/C5JP48 |
| `credit_default.csv` | Default of Credit Card Clients | I-Cheng Yeh | 2009 | https://doi.org/10.24432/C55S3H |
| `liver_disease.csv` | ILPD (Indian Liver Patient Dataset) | Bendi Ramana and N. Venkateswarlu | 2022 | https://doi.org/10.24432/C5D02C |
| `student_dropout.csv` | Predict Students' Dropout and Academic Success | Valentim Realinho, Monica Vieira Martins, Jorge Machado and Luis Baptista | 2021 | https://doi.org/10.24432/C5MC89 |

UCI lists a related paper for the credit data (Yeh and Lien, 2009, Expert Systems with
Applications). The UCI page does not say that citing it is required.

### Changes made and what was verified

The checks below were made on the files in this repository and on the research copies of the
source files, by comparing values and class counts. No dataset was downloaded for this release.

| File | Changes made | What was checked | What is not verified |
|---|---|---|---|
| `car_quality.csv` (1,728 rows, 6 features) | The six categorical attributes are ordinal-encoded: buying and maint low=0, med=1, high=2, vhigh=3; doors 2=0, 3=1, 4=2, 5more=3; persons 2=0, 4=1, more=2; lug_boot small=0, med=1, big=2; safety low=0, med=1, high=2. The target is 1 for the class "good" and 0 for every other class. | The encoding above was checked row by row against the KEEL file `car-good.dat` (1,728 rows). The target has 69 rows of 1 and 1,659 of 0, which matches the 69 "good" cars of the source. | The original conversion script is not preserved. Whether the file was produced from the UCI file or from the KEEL file is not recorded. |
| `credit_default.csv` (6,000 rows, 23 features) | A 6,000-row subsample of the 30,000-row source, the ID column dropped, the "default payment next month" column renamed `target` (1 = default). | Every one of the 6,000 rows was found in the source file by value. The default rate is 0.2212 in the source (6,636 of 30,000) and 0.2212 in the sample (1,327 of 6,000). The sample holds 5,997 distinct rows, because the source itself contains duplicate rows. | How the 6,000 rows were chosen (method, seed, stratification) is not recorded and cannot be reproduced from this repository. The file contains the demographic columns SEX, EDUCATION, MARRIAGE and AGE of the original public release. |
| `liver_disease.csv` (583 rows, 10 features) | Gender recoded to 0 (Female) and 1 (Male). Columns given descriptive names. The class label is recoded: see the next row. | Row by row against the research copy of the source file: all 583 rows agree on the numeric columns and on the gender recoding. | The column names were matched to the source by position, not against the original header text. |
| `liver_disease.csv`, target meaning | Source class 1 (416 records, "liver patient" in the UCI coding) becomes `target` 0. Source class 2 (167 records, "non liver patient") becomes `target` 1. | The sample file has 416 rows of 0 and 167 rows of 1. The published class counts are 416 liver patients and 167 non-patients. The crosstab of source class against sample target is exact (416 and 167, no mixing). | The rare class (`target` = 1) is therefore the group WITHOUT liver disease. The in-app card text for this sample used to say "spot patients with liver disease", the opposite. It is now fixed and reads "find the less common group: records without liver disease". |
| `student_dropout.csv` (4,424 rows, 36 features) | The three-class outcome is binarised: `target` is 1 for "Enrolled" and 0 for "Dropout" and "Graduate". Column names are tidied (spaces and slashes replaced). Two grade columns are rounded to four decimals. | All 4,424 rows are in the same order as the research copy of the source file. The sample has 794 rows of 1, which equals the published "Enrolled" count (Dropout 1,421, Graduate 2,209, Enrolled 794). The crosstab of source outcome against sample target is exact. | The rare class (`target` = 1) is "Enrolled", not "Dropout". The in-app card text used to say "flag students at risk of dropping out", which did not match. It is now fixed and reads "find the less common group: currently enrolled students". The rounding of the two grade columns was recorded by an earlier review pass and was not repeated here. |

## 2. Datasets used by the research runner (not included)

The research runner reads 15 datasets. They are not part of this repository. Download each from
the original provider and place it as described in `README.md`. File hashes are in
`docs/RESULTS_EVIDENCE.md` so a reader can check that a downloaded file is the same one.

| Datasets | Provider | Terms as found |
|---|---|---|
| PSDAS, DCCC, BCW, ESDRP, CB (Sonar), GLASS, ILPD, SEED, WINE | UCI Machine Learning Repository, https://archive.ics.uci.edu/ | CC BY 4.0 on the UCI pages, as read on 4 November 2026 |
| ESR (Epileptic Seizure Recognition) | UCI, derived from recordings of the University of Bonn | Not established. The UCI page and the Bonn data page could not be read (access refused) when this was checked, and one secondary source says the UCI copy was withdrawn at the curators' request, which was not confirmed. Obtain it from the original source and read its terms. |
| YEAST5, YEAST5-ERL, CARGOOD, CARVGOOD, FLARE-F | KEEL imbalanced datasets repository, https://sci2s.ugr.es/keel/imbalanced.php | The KEEL page states no licence or reuse terms. The data are processed variants of UCI datasets (Yeast, Car Evaluation, Solar Flare). Do not assume the UCI licence covers the KEEL-processed files. |

KEEL repository reference: J. Alcala-Fdez et al., "KEEL Data-Mining Software Tool: Data Set
Repository, Integration of Algorithms and Experimental Analysis Framework", Journal of
Multiple-Valued Logic and Soft Computing 17(2-3), 255-287, 2011. The author order differs between
catalogues and was not re-checked against the journal; verify it before citing.

## 3. The CSRBoost paper

Seema Yadav, Dhruvanshu Joshi, Soham Mulye, Labib Asari, Sandeep S. Udmale and Girish P. Bhole,
"CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern
Classification", IEEE Access, volume 13, pages 175078-175093, 2025,
https://doi.org/10.1109/ACCESS.2025.3616207

The author list and bibliographic details were read from the paper's front matter and from the
Crossref record on 4 November 2026; the two agree. The paper states that it is licensed under
CC BY 4.0. It is not redistributed in this repository. This project's code is an independent
implementation of the published recipe, not the authors' code, and it does not reproduce every
detail of the published method (see `docs/IMPLEMENTATION_MAP.md`).

## 4. Software dependencies

Nothing is vendored. Dependencies are installed from PyPI at build or setup time (see
`requirements.txt`) and keep their own licences. Licences below were read from installed package
metadata on 4 November 2026 (the numpy version installed for that check was newer than the
pinned one, so its row describes the installed version).

| Package | Licence |
|---|---|
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause. numpy wheels also bundle the libgfortran and libquadmath runtime libraries, which keep their own terms; nothing is vendored in this repository |
| scikit-learn | BSD-3-Clause |
| imbalanced-learn | MIT |
| openpyxl | MIT |
| xlrd | BSD |
| scipy (used by the research runner) | BSD-3-Clause |

The Docker image is built from `python:3.11-slim`; Python and the Debian packages in the image
carry their own licences and are not stored in this repository.

## 5. Front-end

`frontend/index.html` is a single file with no external script, stylesheet, font or image. Fonts
are system fonts. The file contains two small inline SVG icons. Their origin is not documented
anywhere in the repository; no icon-set name or licence text was found in the file, and it has not
been independently confirmed that neither icon comes from a named icon set.
