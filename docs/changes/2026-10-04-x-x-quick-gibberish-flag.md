# 2026-10-04 — X-X quick gibberish flag

Zadato ručno od vlasnika (van reda, ne preko `next_ticket.sh`; `ps` potvrdio da runner/loop ne rade). Direktno na `main`.

## Šta je promenjeno
Dok je X popup (flag panel) otvoren, taster **X** odmah snima flag kategorije `gibberish` bez komentara i zatvara popup. Pored opcije u popup-u piše `1 · Gibberish [X]`.

Fajl: `web_labeler/static/app.js` (samo JS; `FLAG_CATEGORIES[gibberish].shortcut`, `saveFlag(opts)`, nova grana u `installHotkeys` za `state.flagPanelOpen`). Test: `web_labeler/tests/test_flag_quick_gibberish.py`.

## Ponašanje
- X pokreće prečicu samo kad fokus NIJE u polju za komentar (`isTypingTarget`), bez Ctrl/Meta/Alt i bez auto-repeat (držanje X ne snima slučajno).
- Ide kroz isti `saveFlag()` kao klik na Gibberish + Save, sa praznim komentarom (i ako je postojeći flag imao komentar, prečica šalje prazan). Isti API poziv (`rawFlagSet` / `frameFlagSet`), isti source `manual_goca`, ista kategorija, ista šema sidecara. Nema promene na serveru.
- `flagCanSave` ostaje na snazi: ako je owner već odlučio, prečica ne radi (isto kao klik).
- Esc, Enter, 1-4 i klik na opcije rade kao do sada. K, D, R, B, O i strelice pripadaju review queue-u i ne diraju se.
- Undo je isti: Unflag dugme u popup-u (`removeFlag`), vraća `replaced_auto` ako postoji.

## Gde važi
X popup postoji na dva mesta i oba koriste isti panel, pa prečica važi na oba:
1. Dataset Fixer, RAW pregled (Load raw).
2. Video Labeler (učitan frame).

Review queue nema X popup (samo K/D/R odluke vlasnika), pa se tamo ništa ne menja. Background Labeler nema X flag.

## Testirano
- `node --check app.js` OK.
- `python -m unittest discover -s web_labeler/tests`: 21/21 OK (17 postojećih + 4 nova; novi su statičke provere ožičenja, ne pravi klik u browseru).
- `IntelliCup/tests/data4_audit.py --pool shots --dry-run` na scratch kopiji `raw_review`: ok, errors=0.

## Nije testirano
- Pravi klik/taster u browseru (X pa X na Dataset Fixer RAW i na video frame-u), ni na jednoj instanci.
- Trka: ako se drugi X pritisne pre nego što se učita postojeći flag, odluka o postojećem flag-u se oslanja na server (vraća grešku kod owner odluke).

## Restart
Nije potreban (menjan samo JS, `no-store`; osvežiti stranicu).
