# Data

## CAMELS-US daily streamflow (`CAMELS/`)

Newman et al. (2015) and Addor et al. (2017); dataset doi:10.5065/D6MW2F4D, record https://zenodo.org/records/15529996,
licence CC BY 4.0 (see `CAMELS/DATA_MANIFEST.json` and `CAMELS/zenodo_record.json`). Included: the observed daily
flow of 12 candidate basins (`CAMELS/pilot_observed_streamflow/`), the catchment attributes and the official
documentation (`CAMELS/archive_documentation/`). Observed files have six fields: gauge ID, year, month, day, discharge
(cubic feet per second) and quality flag; -999 marks missing flow.

Selection, fixed before any fit: the lowest gauge identifier in each of the first 12 HUC-02 regions (sorted); basins
with more than 5% missing daily flow in 1981-2014 are dropped (06037500 and 07290650), which leaves ten basins.

## NOAA borehole temperature profiles (`Borehole_NOAA_Huang/`)

Global Borehole Temperature Database for Climate Reconstruction (Huang and Pollack, 1998; 2016 update), NOAA/NCEI
World Data Service for Paleoclimatology, https://www.ncei.noaa.gov/pub/data/paleo/borehole/huang2000/ ; public data,
cite Huang and Pollack (1998) and Huang, Pollack and Shen (2000). Included: the 1012 profiles of the 2016 update
(`huang2016/`), the provider readme (`readme_borehole.txt`) and a manifest with SHA-256 hashes (`DATA_MANIFEST.json`).

## CRUTEM5 (not included)

The borehole site selection uses CRUTEM5 station data (CRUTEM.5.2.0.0, (c) British Crown Copyright, Met Office,
Open Government Licence v3). Download `CRUTEM.5.2.0.0.anomalies.nc` and the station files from
http://www.metoffice.gov.uk/hadobs/crutem5 (147 MB) and place them in `04_DATA/CRUTEM5/`.
