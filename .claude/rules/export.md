# Export pipeline (`export_all()` i naming konvencije)

## Layout

`export_all()` u `web_labeler/server.py:1679-1864` piše, po modelu:

```
<output_base_dir>/<VIDEO_PREFIX>_<MODEL>/images/<BASE>.jpg
<output_base_dir>/<VIDEO_PREFIX>_<MODEL>/labels/<BASE>.txt
<output_base_dir>/<VIDEO_PREFIX>_<MODEL>/data.yaml
<output_base_dir>/<VIDEO_PREFIX>_<MODEL>/image_tags.json   (frame/bbox tag metadata, server.py:1797-1833)
```

- Folder ime: `batch_dirname(video, model)` = `<VIDEO_PREFIX>_<MODEL>` (`naming.py:96-111`),
  npr. `BLAZNAVAC_SANK_LEVO_20251205090046_GLASSES`. Docstring (`naming.py:98-99`):
  *"This is the folder that gets zipped and dropped into the ingestion pipeline."*
- `BASE` = `<TIMESTAMP>_<VIDEO_PREFIX>_f<FRAME>` — GUID iz video imena se uklanja (Windows
  path limit). Ime slike i label fajla je uvek identično (`<base>.jpg` / `<base>.txt`).
- `BAR_COUNTER_INFO` (`SANK_LEVO`/`SANK_DESNO`/`SANK_TOCILICA`) je obavezan token u imenu
  videa; ako nedostaje, export traži izbor od korisnika (`naming.py:29-78`, `server.py:1690-1706`).
- `data.yaml` se kopira iz modela u svaki batch folder (`server.py:1753-1763`).
- Nakon pisanja, svaki model-folder se zipuje preko `_zip_batch` (`server.py:1164-1179`,
  poziv `server.py:1837-1845`) i originalni folder se briše — zip sadrži jedan top-level
  folder sa `images/`, `labels/`, `data.yaml` (docstring `_zip_batch`, `server.py:1169-1170`:
  "pipeline expects a single top-level folder inside the zip").
- `output_base_dir` je konfigurabilan preko `LABELER_OUTPUT_DIR` / `output_dir` u
  `labeler_config.json` (default `~/LabelingToolData/output`, ili `output/` u repo za dev).
  Odredišni `to_ingest/` naziv **nije ovde** — određuje se na IntelliCup strani
  (`ingestion_pipeline`/`run_ingest.py`, vidi `IntelliCup/CLAUDE.md` §3).
- Server drži sve u memoriji — `AppState` se resetuje posle svakog exporta
  (`state.py: reset_video_state`, poziv `server.py:1853`).

## Duplikacija (CB-1-stil)

Labeled-frame export (`server.py:1719-1763`) i background-frame export (`server.py:1766-1795`)
su gotovo identičan blok (image write, label write, `data.yaml` copy) — razlikuju se samo u
sadržaju label fajla. Proveri i background batch export logiku dalje u fajlu (`zip_target`/
`out_root`, oko linije 742-780) za istu šemu nezavisno duplirana. Izmena formata/uslova na
jednom mestu mora se ogledati na svim ostalim.

## LBL-1 — silent except oko data.yaml copy

`except Exception: pass  # ignore copy errors` na `server.py:1762-1763` i `server.py:1793-1794`.
Ako copy `data.yaml`-a ne uspe, export nastavlja bez upozorenja korisniku — batch zip može
završiti u ingestion pipeline-u bez `data.yaml`. Fix: logovati/vratiti warning korisniku
umesto tihog pass-a.

## `_zip_batch` fail-soft

Ako zipovanje batch-a ne uspe, kod loguje warning i i dalje dodaje putanju u `zip_paths`
(`server.py:1836-1845`) — ali to je putanja *foldera*, ne `.zip` fajla. Downstream ingestion
koji očekuje `.zip` može dobiti nepostojeći/pogrešan put. Ne ignoriši ovo pri debagovanju
"missing batch" pritužbi sa ingestion strane.
