# labelingTool

## 1. Šta repo radi

Browser-based (FastAPI + vanilla JS) alat za labeliranje video snimaka za YOLO trening,
za IntelliCup ekosistem (Blaznavac venue). Tri moda: **Labeler** (bbox crtanje, export u
YOLO format), **Dataset Fixer** (korekcija postojećih dataset-ova), **Background Labeler**
(negativni/background uzorci) — plus noviji **Visual Analyzer** tab (pre-label sugestije
preko prave YOLO inference). `README.md` je **zastareo** — ne pominje analyzer, ingestion
ni model-reading; ne veruj mu potpuno dok se ne ažurira.

## 2. Entry points

- `run_web_labeler.py` — pokreće uvicorn (`web_labeler.server:app`), `LABELER_HOST/PORT/OPEN_BROWSER`.
- `web_labeler/server.py` — `create_app()`, ~51 `/api/...` ruta (Labeler, Dataset Fixer,
  Background Labeler, Analyzer proxy). Sav state je in-memory (`AppState`), nema baze.
- `web_labeler/analyzer_worker.py` — subprocess worker, radi pod **odvojenim** interpreterom
  (`/opt/interpreters/INTELLICUP_MODELS/bin/python`, ima ultralytics/torch), poziva se iz
  `analyzer.py`. Detalji: `.claude/rules/analyzer.md`.
- `web_labeler/naming.py` — export imenovanje (video/frame/batch/zip). Detalji: `.claude/rules/export.md`.
- **LEGACY, ne koristiti** — `labelingMvpV6_1.py` (root): stari standalone tkinter app,
  zamenjen web_labeler-om, nije referenciran ni iz jednog run skripta.

## 3. Cross-repo zavisnosti

**Ovaj repo šalje:** `export_all()` (`server.py:1679-1864`) zipuje per-model batch-eve
(`images/`, `labels/`, `data.yaml`) u layout usklađen sa IntelliCup `ingestion_pipeline`
očekivanjem (docstring `naming.py:98-99`). Odredišni folder (`to_ingest/`, po `IntelliCup/CLAUDE.md`
§3) **nije hardkodovan ovde** — `output_base_dir` je konfigurabilan (`LABELER_OUTPUT_DIR`).
Format layout-a je de facto cross-repo ugovor: ne menjaj ga bez provere IntelliCup strane.

**Ovaj repo prima (read-only):** `current.pt` iz `/opt/intellicup/models/<model>/` za
pre-label sugestije u Visual Analyzer tabu — potvrđeno bez ijedne `torch.save`/download
linije u repou. Detalji: `.claude/rules/analyzer.md`.

**Jaka zavisnost (ne samo formatska):** `server.py:359` + `labeler_config.json:5` čitaju
`~/Projects/IntelliCup/utils/blaznavac_article_map.yaml` **direktno sa diska** za mapiranje
naziva artikala u klase — ako se taj fajl pomeri/preimenuje u `IntelliCup` repou, ovaj repo
tiho ne dobija mapiranje (nema fallback/error surface potvrđen).

**Potvrđeno: nema veze sa `intellicup_deep_sort`** (0 referenci na deep_sort/deepsort u repou).

## 4. Rizična mesta

- **CB-1-stil duplikacija**: labeled-frame export (`server.py:1719-1763`) i background-frame
  export (`server.py:1766-1795`) su skoro identičan copy-paste — izmena na jednom mestu
  lako zaboravi drugo. Detalji: `.claude/rules/export.md`.
- **LBL-1**: silent `except Exception: pass` oko `data.yaml` copy (`server.py:1762-1763,
  1793-1794`) — batch može otići u ingestion bez `data.yaml`, bez upozorenja.
- **Mrtav config key**: `raw_base_path` (`labeler_config.json:6`) izgleda povezan sa
  analyzer-om ali se nigde ne čita — pravi key je `ANALYZER_RAW_ROOT`.
- **`_zip_batch` fail-soft** (`server.py:1164-1179`, poziv `:1836-1845`): ako zip padne,
  u `zip_paths` ide *putanja foldera* umesto zipa — downstream ingestion koji očekuje
  `.zip` može dobiti nepostojeći put.

## 5. Pravila pre izmene

- Analyzer je **read-only** prema modelima i raw dataset-u — nijedna izmena ne sme uvesti
  pisanje/brisanje u `/opt/intellicup/models/` ili `/opt/intellicup/datasets/raw/`.
- Export layout (`images/labels/data.yaml` šema, `batch_dirname` konvencija) je cross-repo
  ugovor sa IntelliCup ingestion pipeline-om — svaka izmena formata mora biti eksplicitno
  najavljena/proverena na toj strani pre merge-a, ne samo ovde.
- Kredencijal/secret nađen u fajlu → stop, ne commit-uj, prijavi (`~/Projects/CRITICAL_BUGS.md` protokol).

## 6. Compact Instructions

Kad se kontekst kompaktuje/sumira, sačuvaj eksplicitno:
- Izmenjeni fajlovi ovu sesiju.
- Test/verifikacione komande realno pokrenute (nema pytest suite-a — verifikacija je
  ručni run kroz `run_web_labeler.py` i probni export) i ishod.
- **Da li je export format menjan ovu sesiju** (naming, layout, `data.yaml` sadržaj) — bitno
  jer IntelliCup ingestion pipeline zavisi od tačnog formata.
- Otvorena pitanja/blokeri za sledeću sesiju.
