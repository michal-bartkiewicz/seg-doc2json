# PESEL name lexicon

`pesel_name_lexicon_2026.csv.gz` is a normalized package of the first-name and surname columns from four aggregate datasets published by the Polish Ministry of Digital Affairs on dane.gov.pl. The data date is 2026-01-20. The portal API reports `CC0 1.0` for both source datasets.

- First names: dataset 1501, resources 1159637 (male) and 1159639 (female).
- Surnames: dataset 1681, resources 1148808 (male) and 1148811 (female).
- Source catalog: https://dane.gov.pl/pl/dataset/1501 and https://dane.gov.pl/pl/dataset/1681
- Metadata API: https://api.dane.gov.pl/1.4/datasets/1501 and https://api.dane.gov.pl/1.4/datasets/1681

The package keeps only distinct names and a `given`/`surname` category; it drops counts and gender. These are aggregate name lists, not person records. A dictionary hit is only a weak linguistic clue and does not establish a person's identity.
