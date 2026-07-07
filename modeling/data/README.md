# Lake h5 inputs (read-only)

ICESat-2 lake depth files from `detect_lakes.py` are stored at the **repo root**:

```
../detection_out_data/lake_*.h5
```

They are not duplicated under `modeling/` because the same files are used by:

- `detect_lakes.py` / `run_sw_lake_detection.py`
- `rename_lakes_and_stats.py`
- `make_quicklook_sw.py`
- `modeling/scripts/extract_is2_s2_pairs.py`

To point extraction at a different folder:

```bash
python modeling/scripts/extract_is2_s2_pairs.py --data-dir path/to/lake_h5
```
