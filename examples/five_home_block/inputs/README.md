# Inputs

## EV mobility

`ev/NHTS_tripv2pub.csv` is the renamed, byte-identical NHTS input formerly
called tripv2pub.csv; cleaning remains in the loader.
The Audi 20-vehicle and 100-vehicle exports are available as CSV.gz and Parquet.
Adapter tests cover both formats, so these are retained rather than pruned as
duplicates. See the [unified pipeline](../../single_home_hems/docs/pipeline.md)
for source selection, replay coverage and distance-based driving energy.
Review dataset redistribution rights before publishing; no source permission
or license is inferred from its presence in this working directory.

## Building and weather

`resstock_2024_2_up11/two_story_unconditioned_basement/` contains the active
ResStock 2024.2 TMY3 upgrade 11 packages selected for the five-home block. Each
home has two conditioned floors above grade and a separate unconditioned
basement. Each directory retains the original ZIP plus its `home.xml`, `in.osm`,
and `in.schedules.csv` files. The preceding heated-basement cohort remains in
the parent directory for provenance but is no longer referenced by the config.

`weather/G1801570.epw` is the official ResStock TMY3 county weather file for
Tippecanoe County, Indiana (FIPS 18157), downloaded from the NLR BuildStock
county weather archive. It is shared by all five homes so neighborhood loads
remain weather-aligned.

To change an envelope or schedule, place the replacement files here and update
the repository-relative paths in `config/homes.json`. Change the common weather
path in `config/neighborhood.json`.
