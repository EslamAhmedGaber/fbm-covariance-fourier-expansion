# Reproducing the new applications (design bracket, borehole design, real borehole data)

Run commands from the repository root unless a `cd` is shown. The new applications used Python 3.12.4; see `requirements-tested.txt`. Limit BLAS threads (for example to 4) when running many small design problems.

## External data (source snapshot: 2026-10-30)

| Data | Location | Provenance |
|---|---|---|
| NOAA/NCEI Huang-Pollack borehole database, `huang2016-*.txt` (1,012 sites) + readme | `04_DATA/Borehole_NOAA_Huang/` | `tools/fetch_data.py --dataset noaa`; manifest `DATA_MANIFEST.json` (SHA-256 per file) |
| CRUTEM.5.2.0.0 gridded land anomalies (5 deg, monthly, vs 1961-1990) | `04_DATA/CRUTEM5/CRUTEM.5.2.0.0.anomalies.nc` | Met Office, OGL v3; sha256 `2540836b4a22ba5d8dfecf7877173357ac37bce3cfddd8b37b6b5569d1e231da` |
| CRUTEM.5.2.0.0 station files (11,865 stations, monthly 2 m air temperature) | `04_DATA/CRUTEM5/stations/CRUTEM.5.2.0.0.station_files.zip` (read in place, not extracted) | Met Office, OGL v3; sha256 `f3f9c81da9b2b167f6caf99634f3ebd0d78f7f92e4fab4d87255575a3d40df90`; manifest `04_DATA/CRUTEM5/DATA_MANIFEST.json`; derived cache `04_DATA/CRUTEM5/derived/crutem5_station_annual.npz` (rebuilt automatically if absent) |
| JMA published annual means, Tokyo 47662, Osaka 47772, Nagoya 47636 (selected years) | embedded in `code/applications/app7_jma_check.py` with source URLs | read from data.jma.go.jp on 2026-10-30; transcription checked against CRUTEM5 where CRUTEM5 is unadjusted (within 0.05 K every year). JMA Public Data License (Version 1.0), CC BY 4.0 compatible, attribution "Source: Japan Meteorological Agency website". |

Citations: Huang & Pollack (1998) database; Huang, Pollack & Shen (2000) Nature 403:756-758; Osborn et al. (2021)
JGR Atmospheres 126, e2019JD032352 (CRUTEM5). Acknowledgement text required by the CRUTEM5 licence:
"CRUTEM.5.2.0.0 data were obtained from http://www.metoffice.gov.uk/hadobs/crutem5 on 30 October 2026 and are (c)
British Crown Copyright, Met Office, provided under an Open Government License".

## New code

| File | Role |
|---|---|
| `code/covfourier/design.py` | `InfoDesign` (bracket, greedy, Frank-Wolfe lower bound), `DenseGreedy` (exact pivoted-Cholesky greedy), `BlockInfoDesign`, `exact_variances` |
| `code/covfourier/khref.py` | independent composite quadrature of the covariance norm (reference; dense-GP covariance builder) |
| `code/covfourier/heat.py` | half-space heat kernels, epoch-mean and heat-content integrands, `project_samples` (the paper's per-input rule on samples) |
| `code/tests/test_design.py` | 5 tests (bracket on all small designs, FW bound vs brute force, dense vs information greedy, block design, quadrature closed forms) |
| `code/applications/app6_design_bracket_check.py` | E0: exhaustive bracket check with window sensors (closed-form exact GP) |
| `code/applications/app6_heat_kernel_check.py` | E1: heat kernels through the interface vs the quadrature reference |
| `code/applications/borehole_problem.py` | shared builder (one basis, two projectors, targets) |
| `code/applications/app6_borehole_design.py` | E2: certified re-logging design, baselines, exact verification |
| `code/applications/app6_design_scaling.py` | E2s: cost against the number of candidates (expansion vs product-quadrature dense alternatives) |
| `code/applications/borehole_data.py` | parser, block attribution rule, profile |
| `code/applications/app7_borehole_realdata.py` | E3a v1 (white noise only; kept as a documented adverse step) |
| `code/applications/app7_borehole_realdata_v2.py` | E3a v2 (Brownian-in-depth geological term) |
| `code/applications/app7_validation_summary.py` | calibration against CRUTEM5, sites and countries |
| `code/applications/app7_relog_forecast.py` | E3b: forecast of a later re-log from an earlier log |
| `code/applications/app7_site_history.py` | example posterior histories |
| `code/applications/app7_borehole_regional.py` | regional model (shared climate + local paths + geology), country histories |
| `code/applications/app7_network_design.py` | E3c real-network design (adverse; not in the manuscript) |
| `code/applications/app7_borehole_regional_adv.py` | groundwater-flow curvature term (pooled and per-country scale), sensitivity |
| `code/applications/app7_station_validation.py` | station archive reader/cache; nearest-station target (rejected as a target) |
| `code/applications/app7_target_error.py` | error of the instrumental target from the station archive; country and site comparison; sensitivity; China split |
| `code/applications/app7_jma_check.py` | JMA published series vs CRUTEM5 versions (Tokyo, Osaka, Nagoya) |
| `figures/src/make_figures_design.py` | Figure 9 |
| `figures/src/make_figure_realdata.py` | Figure 10 |

## Commands (in this order; times measured on the shared 12-core machine under load)

```
cd code/tests && python test_covfourier.py && python test_design.py
cd ../applications
python app6_design_bracket_check.py          # ~6 min single-threaded
python app6_heat_kernel_check.py             # ~8 min
python app6_borehole_design.py               # ~40 min (four bases, relaxations, quadrature verification)
python ../../tools/run_design_scaling.py       # scaling configurations, one process each
python app7_borehole_realdata_v2.py          # ~6 x 5 min; caches coordinates in results/cache
python app7_validation_summary.py A7_borehole_realdata_v2.json A7_validation_summary_v2.json
python app7_relog_forecast.py
python app7_site_history.py
python app7_borehole_regional.py 0.5         # regional model, 13 countries (~3 min)
python -c "import app7_borehole_regional as R; R.country_history('Canada'); R.country_history('United States Of America')"
python app7_network_design.py                # E3c, adverse, not in the manuscript
python app7_borehole_regional_adv.py 0.5     # flow curvature, pooled scale (~4 min)
python app7_borehole_regional_adv.py by_country 0.5
python app7_station_validation.py base       # first run builds the station cache (~2 min)
python app7_target_error.py 0.5              # target error, country/site comparison, sensitivity, China split (~10 min)
python app7_jma_check.py
cd ../.. && python figures/src/make_figures_design.py fig9 && python figures/src/make_figure_realdata.py
```

The private manuscript audits compare quoted text with stored results. Their release-time outcomes are recorded in `RELEASE_VALIDATION.md`; the unpublished manuscript is not part of this software repository.

## Figure-to-data mapping

| Panel | Script | Data |
|---|---|---|
| Fig. 9a | `figures/src/make_figures_design.py fig9` | `results/A6_borehole_design.json` (main.fw, main.baselines, refinement n=256, verification) |
| Fig. 9b | same | `results/A6_design_scoremap.json`, `A6_borehole_design.json` (main.greedy_z/t) |
| Fig. 9c | same | `A6_borehole_design.json` (refinement n = 64..512, main.fw) |
| Fig. 9d | same | `results/A6_design_scaling.jsonl` |
| Fig. 10a,b | `figures/src/make_figure_realdata.py` | `results/A7_regional_history_Canada.json`, `..._United_States_Of_America.json` |
| Fig. 10c | same | `results/A7_borehole_regional.json` (by_H 0.5 countries), `A7_validation_summary_v2.json` (database reconstructions), `A7_target_error.json` (horizontal bars: target error) |
| Fig. 10d | same | `results/A7_relog_forecast.json` |
Plotted numbers are also written to `figures/data/fig9.json` and `fig10.json`.

## Seeds and configurations

E0 seed 7 (window draws); E2 baselines seed 11 (random designs), verification random designs seed 12; E3 split seed 3
(345 training sites); E3c random baseline seed 5. Configurations are stored in the `config` block of every result file.
The result files retain the model-development sequence (v1 white noise kept as adverse step,
`A7_borehole_realdata_v1_partial.json`; v2 geological term; window sensitivity `A7_borehole_realdata_v2_window500.json`;
regional model `A7_borehole_regional.json`).
