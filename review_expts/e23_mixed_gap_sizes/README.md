# E23 - Mixed gap sizes (E6 "native" pattern)

Same setup as E22, but gap lengths are drawn from the record's own native missing runs (2-365 days) until 10% of each station is masked. This is the same logic as `common/data.py` `make_gaps(pattern='native')`.
- Arms: vanilla, `pimf` (final), `pimf_old` (v5/v6 paper). The pickles call them `vanilla`, `pimfA`, `pimf`; `analyse.py` maps the names.
- Seeds: 42 and 1. There is no PIMF-L arm.
- The realised masked fraction is below 10%, because `placed += L` also counts cells that were already missing.

## Run (uses E22's script)
```
python ../e22_anchor_n_placebo/run_cell.py --root <repo> --data <repo>/mahanadi_data --pattern native --seed-block --seed 42 --gap 0 --arms vanilla,pimf,pimf_old --njobs 10 --out results/cells/cell_42_native.pkl
python analyse.py results/cells > results/summary.txt
```

## Findings
2 seeds; 92% of scored days (7,319 of 7,936) lie in gaps over 30 days, so this mostly tests long gaps.
- **Volume bias:** old PIMF +10 to +14%; PIMF +1 to +3%; vanilla -0.2 to +2.2%.
- **Old PIMF vs vanilla:** it beats vanilla in only 27% of station-seed cases (p = 0.012), so it is worse.
- **PIMF vs old PIMF:** PIMF is better in 69% of cases (p = 0.006).
- **PIMF vs vanilla:** PIMF is better in 65% of cases, but **p = 0.21, so not shown to beat vanilla**. With 2 seeds and 13 stations the cases are not independent either. The 95% figure for 31-100-day gaps is a post-hoc subgroup and should not be cited as evidence.
- Too few short-gap days (66 in 2-7 d gaps, 551 in 8-30 d) to judge short gaps.
