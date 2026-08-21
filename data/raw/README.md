# Raw data

This folder is intentionally empty in version control (`.gitignore`d) — the
raw files are too large / not ours to redistribute under a separate license.

## Download

Get the two raw source files from the Environmental Data Initiative (EDI)
repository, data package **`knb-lter-and.5216`**, released under a
**CC BY 4.0** license by the H.J. Andrews Experimental Forest LTER program:

- Data package page: https://andlter.forestry.oregonstate.edu/data/abstract.aspx?dbcode=SA026

Download and place the following files here:

```
data/raw/SA02601_v6.csv   # raw visitation events   (74,942 rows x 26 cols)
data/raw/SA02602_v5.csv   # raw flowering phenology  (23,642 rows x 16 cols)
```

## Expected schema

`SA02601_v6.csv` (visitation): `DBCODE, ENTITY, COMPLEX, MEADOW, PLOT_ID, YEAR,
SAMPLEDATE (DD-MM-YYYY), WATCH, OBSERVER, PLOT, START_TIME, END_TIME, MINUTE,
CLOUDS, WIND, TEMP, PPI_STATUS, NO_INT, PLTSP_CODE, PLTSP_NAME, VISSP_CODE,
VISSP_NAME, VISSP_TYPE, REF_NO, VISSP_NO, QC_NOTES`

`SA02602_v5.csv` (flowering phenology): `DBCODE, ENTITY, COMPLEX, MEADOW,
PLOT_ID, YEAR, SAMPLEDATE (YYYY-MM-DD), WATCH, OBSERVER, PLOT, FLW_STATUS,
PLTSP_CODE, PLTSP_NAME, NO_STALK, NO_FLWS, QC_NOTES`

**Note the different `SAMPLEDATE` formats between the two files** — this is
handled explicitly by `pp_tgnn.pipeline.harmonize_raw_data()` (Algorithm 1 in
the paper); see the main README for details.

If your local copy of these files uses different column names or a different
release version, adjust `PipelineConfig` in `src/pp_tgnn/config.py` (or pass a
custom `PipelineConfig(...)` instance to the pipeline functions) rather than
editing the pipeline code.

## Attribution

When reusing the raw or processed data, please cite:

> H.J. Andrews Experimental Forest LTER, "Plant-pollinator interactions in
> Willamette National Forest meadows," Environmental Data Initiative, data
> package knb-lter-and.5216.
