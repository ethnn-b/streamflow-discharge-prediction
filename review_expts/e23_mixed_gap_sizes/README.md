# E23 - Mixed gap sizes (E6 "native" pattern)

Same setup as E22, but gap lengths are drawn from the record's own native missing runs (2-365 days) until 10% of
each station is masked: the same logic as `common/data.py` `make_gaps(pattern='native')`. Arms vanilla, pimf, pimfA; seeds 42 and 1.

## Run (uses E22's script)
```
python ../e22_anchor_n_placebo/run_cell.py --root <new-pimf checkout> --data <repo>/mahanadi_data --pattern native --seed 42 --gap 0 --arms vanilla,pimf,pimfA --njobs 10 --out results/cells/cell_42_native.pkl
python analyse.py results/cells > results/summary.txt
```

## Findings (2 seeds; 94% of scored days lie in gaps over 30 days)
- Bias: old PIMF +10 to +14%, PIMF-A +1 to +3%, vanilla about 0 to +2%.
- Per station: old PIMF beats vanilla in 27% of cases (p = 0.012); PIMF-A in 65% (p = 0.21); PIMF-A beats old PIMF in 69% (p = 0.006).
- 31-100-day gaps: PIMF-A beats vanilla in 95% of station-seeds (p < 0.001). Over 100 days: level with vanilla. Too few short-gap days to judge.
