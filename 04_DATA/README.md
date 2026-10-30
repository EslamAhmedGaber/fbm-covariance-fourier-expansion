# External data

Raw NOAA and CRUTEM5 data are downloaded from their providers, not redistributed in the Git repository. The manifests record the exact URLs, sizes and SHA-256 hashes of the 30 October 2026 source snapshot.

```sh
python tools/fetch_data.py --dataset noaa
python tools/fetch_data.py --dataset crutem5
python tools/fetch_data.py --dataset all --verify-only
```

For a small connectivity/checksum test, use `--dataset noaa --limit 2`. The full NOAA set has 1,012 profiles plus a readme; CRUTEM5 has a roughly 13.5 MB grid file and 138 MB station archive. Derived caches are rebuilt by the experiment scripts.

The loader discovers this repository's `04_DATA` folder. To reuse an existing data tree, set `BOREHOLE_ROOT` to the directory **containing** `04_DATA`; do not set it to `04_DATA` itself. In PowerShell: `$env:BOREHOLE_ROOT='D:\your-data-workspace'`.

## Sources and attribution

- NOAA/NCEI Huang–Pollack borehole database, 2016 files: [source directory](https://www.ncei.noaa.gov/pub/data/paleo/borehole/huang2000/). Cite Huang and Pollack (1998), *Global borehole temperature database for climate reconstruction*, Data Contribution Series 1998-044, and Huang, Pollack and Shen (2000), *Nature* 403, 756–758. Read the downloaded provider readme.
- [CRUTEM5](https://www.metoffice.gov.uk/hadobs/crutem5/), release CRUTEM.5.2.0.0; cite Osborn et al. (2021), *JGR Atmospheres* 126, e2019JD032352. CRUTEM.5.2.0.0 data were obtained from http://www.metoffice.gov.uk/hadobs/crutem5 on 30 October 2026 and are (c) British Crown Copyright, Met Office, provided under the Open Government Licence v3.
- Selected published annual station means are transcribed in `code/applications/app7_jma_check.py` with their source URLs. Source: Japan Meteorological Agency website. JMA content is subject to the [JMA usage terms](https://www.jma.go.jp/jma/en/copyright.html), including the applicable Public Data License and any stated exceptions. The script documents the Tokyo, Osaka and Nagoya series used.

Downloading verifies bytes against the recorded snapshot. If a provider replaces a file, the script stops on a hash mismatch rather than silently substituting newer data.
