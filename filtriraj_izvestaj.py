import csv
import glob
import os
from pathlib import Path
from collections import defaultdict

desktop = Path.home() / "Desktop" / "Labeler"
input_dir = desktop / "Izvestaji"
output_dir = desktop / "Pica_po_vremenima"
targets_file = desktop / "pica_za_trazenje.txt"

output_dir.mkdir(parents=True, exist_ok=True)

# Učitaj target pića
if not targets_file.exists():
    print(f"⚠️ Nema fajla {targets_file.name}. Dodaj pića koja želiš da filtriraš.")
    exit(1)

targets = [line.strip().upper() for line in targets_file.read_text(encoding="utf-8").splitlines() if line.strip()]
if not targets:
    print("⚠️ pica_za_trazenje.txt je prazan — dodaj nazive pića.")
    exit(1)

# Pronađi sve csv fajlove
input_files = glob.glob(str(input_dir / "*.csv"))
if not input_files:
    print(f"⚠️ Nema CSV fajlova u {input_dir}")
    exit(1)

summary = []
per_article = defaultdict(list)

for f in input_files:
    with open(f, encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for row in reader:
            # Nađi kolone — prilagodi ako treba (ovde primer iz tvog CSV-a)
            # Pretpostavka: datum je na poziciji 16, artikal na ~21 (proveri u tvom CSV-u)
            if len(row) < 25:
                continue
            # U tvom CSV timestamp izgleda "2024.12.04 09:32"
            timestamp = next((c for c in row if c.strip().startswith("202")), "")
            artikal = row[20].strip().upper() if len(row) > 20 else ""
            if artikal and any(t in artikal for t in targets):
                # Dodaj u summary
                summary.append([artikal, timestamp, os.path.basename(f)])
                per_article[artikal].append([timestamp, os.path.basename(f)])

# Snimi summary fajl
with open(output_dir / "sva_pica_sva_vremena.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Artikal", "Vreme", "Fajl"])
    writer.writerows(summary)

# Snimi pojedinačne fajlove po artiklima
for artikal, rows in per_article.items():
    fname = f"{artikal}.csv".replace("/", "_")
    with open(output_dir / fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Vreme", "Fajl"])
        writer.writerows(rows)

print(f"✅ Završeno! Generisano {len(per_article)} CSV fajlova i sva_pica_sva_vremena.csv u {output_dir}")
