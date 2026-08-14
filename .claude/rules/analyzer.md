# Visual Analyzer (`analyzer.py` / `analyzer_worker.py`)

## Arhitektura

Analyzer namerno pokreće inference u **posebnom subprocess-u, pod drugim Python
interpreterom** — glavni labeler interpreter nema ultralytics/torch (docstring
`analyzer.py:1-6`):

```
server.py (POST /api/analyzer/classify, server.py:790-803, poziv na :800)
  → analyzer.py: classify_image() (analyzer.py:86-124)
    → subprocess: analyzer_worker.py classify --model-pt <pt> --image <tmp>
      pod /opt/interpreters/INTELLICUP_MODELS/bin/python (ANALYZER_MODELS_PYTHON, server.py:365-372)
      → analyzer_worker.py:_load_model() (linije 20-22): YOLO(model_pt), ultralytics
      → JSON-lines preko stdout nazad ka analyzer.py
```

Isti mehanizam koristi `start_overlap_report()` (`analyzer.py:207-232`) za "overlap report".

## Konfiguracija (defaults, `server.py:365-372`)

- `models_root` = `/opt/intellicup/models` (`ANALYZER_MODELS_ROOT`)
- `raw_root` = `/opt/intellicup/datasets/raw/blaznavac` (`ANALYZER_RAW_ROOT`)
- `models_python` = `/opt/interpreters/INTELLICUP_MODELS/bin/python` (`ANALYZER_MODELS_PYTHON`)

`analyzer.py:29-30` ponavlja iste putanje u komentarima.

## Read-only garancija (potvrđeno)

- `get_available_models()` (`analyzer.py:49-59`) samo listira foldere u `_models_root` koji
  sadrže `current.pt`.
- `_model_pt()` (`analyzer.py:62-66`) vraća `<models_root>/<model>/current.pt` ako postoji —
  samo čita.
- `get_class_examples()` (`analyzer.py:129-198`) čita slike/label-ove iz `raw_root` — samo čita.
- Grep za `torch.save`, `.save(`, `download`, `urlretrieve`, `requests.get/post` kroz ceo
  `web_labeler/` = **0 rezultata**. Nema training/fine-tuning/model-update/download logike
  igde u repou. `configure()` samo postavlja putanju pri startu servera, nikad ne piše fajl
  unutar `_models_root` ili `raw_root`.
- **Pravilo**: svaka buduća izmena analyzer koda koja bi uvela pisanje u ova dva foldera je
  van scope-a ovog repoa — takva logika pripada `IntelliCup/training_pipeline` (trening) ili
  `promote_model.py` (promocija), ne ovde.

## Poznato ograničenje

`is_available()` (`analyzer.py:41-46`) proverava samo da folder postoji, ne validnost ili
verziju modela unutar njega — ako se model na `/opt/intellicup/models/<model>/` promeni na
nekompatibilnu verziju, analyzer to neće detektovati unapred, pašće tek pri `classify_image()`
pozivu (subprocess timeout=60, `analyzer.py:112-117`).

## Mrtav config key

`raw_base_path` u `labeler_config.json:6` cilja istu putanju kao `ANALYZER_RAW_ROOT`, ali se
nigde u kodu ne čita (samo `analyzer_raw_root` key je wired). Ne pretpostavljaj da menjanje
`raw_base_path` u configu ima efekat.
