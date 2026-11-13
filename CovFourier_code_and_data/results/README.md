# Stored results

One JSON file per experiment and setting (`E1_<kind>_H<H>.json`, `E3_*`, `E4_*`, `E5_learn_hurst.json`,
`E5b_enclosure.json`, `E5c_enclosure.json`, `E6_scale.json`, `E7_real_rivers.json`, `E7b_per_basin.json`,
`E7c_per_basin_centre_train.json`, `E8_real_boreholes*.json` (H 0.50-0.75 in the first file, 0.85 and 0.95 in their
own files; `ref_*` fields are the reference values stored with the v1.3.0 model selection), `E10_stress.json`,
`E11_precision.json`, `E12_cost_model.json`, `F1_idea.json`).
- `raw/` - per-data-set arrays (E1 gaps, E5/E5b/E5c profiles).
Kinds: P point values, W window averages, S heat-conduction kernels. Units: nats; times in seconds.
