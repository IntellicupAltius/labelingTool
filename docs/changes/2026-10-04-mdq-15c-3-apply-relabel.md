# 2026-10-04 — MDQ-15c-3: apply a corrected relabel batch to RAW

Izvršeno preko reda na zahtev vlasnika, ručno (ne preko `next_ticket.sh`). Commit: vidi `git log` ("MDQ-15c-3", `main`).
Nova skripta `web_labeler/apply_relabel.py` (odvojena; `archive_raw.py` NIJE menjan, koristi njegov raspored arhive,
`_unique_dest` i podrazumevane putanje). Čita manifest formata v1 iz `2026-10-01-mdq-15c-2-relabel-batch-export.md`.
Ovim zadatkom `--apply` NIJE pokretan nad pravim RAW-om (ne postoji ispravljen paket).

## Komande
```
PY=/opt/interpreters/INTELLICUP_LABELING_TOOL/bin/python
$PY web_labeler/apply_relabel.py                                   # dry-run, svi otvoreni batch-evi
$PY web_labeler/apply_relabel.py --batch shots/RELABEL_shots_<ts>  # dry-run jednog batch-a
$PY web_labeler/apply_relabel.py --batch shots/RELABEL_shots_<ts> --apply
$PY web_labeler/apply_relabel.py --rollback <stamp>_relabel [--apply]   # povratak cele primene (dry-run bez --apply)
```
`--execute` je alias za `--apply`. Opcije: `--box-tol-px` (default 2.0), `--raw-base-path`, `--review-dir`, `--batches-dir`, `--archive-root`, `--ingest-root`.

## Redosled po stavci
1. već urađeno (`apply_state.json` u batch folderu) → preskoči (idempotentno; ako fali oznaka u sidecaru, dopuni je).
2. ispravljena kopija: `labels/<stem>.txt` postoji (prazan label je legitiman = background), parsira, klase u RAW `data.yaml`, koordinate u [0,1]; `images/<ime>` sha256 == `relabel_snapshot.image_sha256`.
3. original u RAW-u odgovara snapshot-u: SVE kopije imena u svim class folderima (i novopridošle) imaju snapshot sha slike i labela (nedostajući label = prazan), skup kopija == `raw_paths` iz manifesta; inače `stale` (zastarela), preskoči.
4. "Nije ispravljeno": poređenje po BOKSOVIMA (Dataset Fixer Save re-kvantizuje, ~1 px). Isti broj boksova, ista klasa po boksu, svaki ugao (x1,y1,x2,y2 u pikselima) unutar **`--box-tol-px` = 2.0 px**. Bez razlike preko tolerancije → NE primenjuje se, sidecar `relabel_result.status = "unchanged"`.
5. arhiva: SVE kopije (slika + label, svaki class folder) se PREMEŠTAJU u `raw_archive/<stamp>_relabel/<pool>/{images,labels}/<class>/` (nikad brisanje) PRE ingest-a.
6. ubacivanje preko ingest distributora (`ingestion_pipeline.distributor.distribute_samples`, `RAW_BASE` preusmeren na RAW iz argumenta): slika + isti label u class folder SVAKE klase iz ispravljenog labela (`background` ako je prazan).
7. provera: u svakom ciljnom folderu slika (sha == snapshot) i label (sha == ispravljen) stvarno postoje, distributor nije prijavio `conflicts_existing_diff`, nema viška kopija.
8. bilo koji pad u 5-7 → rollback te stavke (ubačeno ide u `<arhiva>/_rolled_back/`, originali nazad), prijava; druge stavke idu dalje.

**Odstupanje od teksta zadatka (namerno):** zadatak kaže "u SVAKI folder gde je postojala kopija". Ingest raspoređuje po klasama iz ispravljenog labela, što je RAW invarijanta (sve kopije imaju sve labele). Ako ispravka doda ili ukloni klasu, folderi se razlikuju od originalnih; plan to ispisuje ("folders differ from the original"). Ako se skup klasa ne menja, folderi su isti kao pre.

## Evidencija primene
- `<raw_archive>/<stamp>_relabel/apply_manifest.json`: po batch-u i stavci arhivirane putanje (from/to/sha256), ubačene putanje (+sha256), ingest index, tolerancija, status (`applied`/`unchanged`/`failed`/`rolled_back`). Ažurira se posle svake stavke.
- Batch folder: `apply_state.json` (stavka → status/run) i `manifest.json.status = "applied"` kad je SVAKA stavka rešena (applied/unchanged); inače ostaje `open` (zastarele/neispravne stavke ostaju vidljive i batch ostaje blokiran za re-export). Posle rollback-a status se vraća na `open`.
- Ingest index `raw/<pool>/_ingest_index/<batch_id>__apply__<run>__<stem>.json` (običan ingest artefakt); rollback ga premešta u `_rolled_back/`.
- Rollback (`--rollback <stamp>`): proverava da su ubačeni fajlovi još tačno ono što je upisano i da su originalna mesta slobodna, pa ubačeno premešta u `_rolled_back/`, originale vraća, briše `relabel_result`, čisti `apply_state`, vraća batch na `open`.

## Sidecar (compat)
Novo OPCIONO polje na unosu: `relabel_result: {status: "applied"|"unchanged", at, batch_id, apply_run}`; piše se preko `FlagStore.set_relabel_result()` (nova metoda; `clear_relabel_result()` za rollback). Unos, `owner_decision` i `relabel_snapshot` se ne diraju, `schema_version` ostaje 1, `FlagStore(...)` i `add_auto_flag()` potpisi nepromenjeni, stari sidecari se čitaju bez migracije. `IntelliCup/tests/data4_audit.py --pool shots --dry-run` (INTELLICUP_TRACKING interpreter) na scratch kopiji sidecara sa `relabel_result`: ok, errors=0.

## Restart servera
NE treba. `server.py` čita sidecar sa diska na svaki zahtev (`flag_store.list_all()` u `/api/review/queue`, `get()` u ostalim rutama), bez keša; slika u review queue-u ima keš ključan po (putanja, mtime, širina). Server ne importuje `apply_relabel.py`. Posledice za UI: (a) UI ne prikazuje `relabel_result` (polje stiže u `entry`, ali nema prikaza); (b) ako ispravka ukloni klasu, ključ `<stara klasa>/<ime>` više nema sliku u RAW-u i kartica se prikaže sa `raw_exists: false`.

## Testovi (`web_labeler/tests/test_apply_relabel.py`, 13 testova, samo temp direktorijumi; ukupno 34/34 OK)
Slika u jednom folderu (+ nova klasa → drugi folder), slika u dva foldera, uklonjena klasa, zastarela (label promenjen; nova kopija se pojavila), ispravljena slika izmenjena / label nedostaje / loša klasa, nije ispravljeno (Save drift) + stvarna ispravka u istom batch-u, dry-run ne menja ništa, tolerancija kao parametar, pad posle `archive`/`ingest`/`verify` sa rollback-om (RAW hash identičan, druga stavka prolazi), idempotentan ponovni run (+ popravka izgubljene sidecar oznake), rollback komanda (+ ponovna primena), stari sidecar bez migracije, regresija `archive_raw.py` (approve_delete premešta par, `relabel` ne). Pravi RAW (`mtime` image/label direktorijuma), `raw_review/` md5 i `relabel_batches/` identični pre i posle. Nije testirano: stvarni `--apply` nad pravim RAW-om, UI prikaz.
