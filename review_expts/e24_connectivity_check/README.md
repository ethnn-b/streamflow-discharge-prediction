# E24 - Empirical check of the Mahanadi connectivity matrix

For each (target, contributor) pair: the Spearman correlation of Q_target(t) with Q_contributor(t - k), k = -3..6, over 1971-2017.
Also covers candidate pairs that are not in the matrix, and each station's record span. Run from the repo root:
`python review_expts/e24_connectivity_check/run.py > review_expts/e24_connectivity_check/results/summary.txt`

## Findings
- All 29 listed pairs peak at a lag of 0-2 days (r = 0.71-0.95), consistent in direction.
- Row V10 is labelled Paramanpur but carries Padampur's coordinates, and neither gauge drains into basantpur. The link has been removed from the CSV.
- Missing real pairs: kesinga -> kantamal (r = 0.958, the strongest in the data), and kantamal/salebhata -> tikarapara.
- boudh, kelo, padampur and paramanpur have no data in 1980-1990.
- Kurubhata, which joins below basantpur, still correlates with it at r = 0.89: correlation alone cannot identify the topology, hence the placebo arm in E22.
