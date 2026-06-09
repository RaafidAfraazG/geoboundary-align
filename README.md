# GeoBoundary Align

A geospatial boundary alignment solution that corrects land plot polygons using satellite imagery
and optional boundary hints.

## Problem

Official land plot boundaries are slightly misaligned with real-world fields visible in satellite
imagery. The goal is to output corrected polygons when confident and flag uncertain plots.

## Method

- Loads `input.geojson`, `imagery.tif`, and optional `boundaries.tif`.
- Starts from the starter/global-shift baseline.
- Searches small x/y shifts around each plot.
- Uses boundary raster alignment as the main scoring signal.
- Penalizes large or weak shifts.
- Marks plots as `corrected` only when alignment confidence is strong.
- Flags uncertain plots and keeps original geometry.

`quickstart.py` remains available as the starter baseline and reference workflow. The final
solution is `run_solution.py`.

## Folder Structure

```text
.
|-- bhume/                         # Starter loading, geometry, baseline, and scoring helpers
|-- data/
|   `-- 34855_vadnerbhairav_chandavad_nashik/
|       |-- input.geojson
|       |-- imagery.tif
|       |-- boundaries.tif
|       |-- example_truths.geojson
|       `-- predictions.geojson
|-- transcripts/
|   `-- README.md                  # AI transcript links
|-- quickstart.py                  # Starter baseline only
|-- run_solution.py                # Final boundary alignment solution
|-- CONTRACT.md
`-- README.md
```

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install geopandas rasterio shapely numpy scipy pillow
```

## Run

```powershell
python run_solution.py data/34855_vadnerbhairav_chandavad_nashik
```

## Output

```text
data/34855_vadnerbhairav_chandavad_nashik/predictions.geojson
```

The output is a GeoJSON `FeatureCollection` with the required fields:
`plot_number`, `status`, `confidence`, `method_note`, and `geometry`.

## Current Validation Result

On `data/34855_vadnerbhairav_chandavad_nashik`:

- 2457 plots processed
- 555 corrected
- 1902 flagged
- median IoU pred = 0.845 vs official = 0.612
- improvement = 0.201
- median centroid error = 5.165 m
- Spearman(conf, IoU) = 0.949

## Transcript

Transcript links are listed in
`transcripts/README.md`.
