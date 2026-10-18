"""Check that data/raw/transfermarkt_davidcariboo.zip is the dataset version used in the paper.

Compares SHA-256 checksums of the zip and of every CSV inside it with data/raw_checksums.sha256.
  python scripts/verify_data.py           check (exit code 0 = same version, 1 = different or missing)
  python scripts/verify_data.py --write   (re)create the checksum file from the current zip
"""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "raw" / "transfermarkt_davidcariboo.zip"
SUMS = ROOT / "data" / "raw_checksums.sha256"
USED = {"players.csv", "player_valuations.csv", "appearances.csv", "transfers.csv", "clubs.csv",
        "competitions.csv", "games.csv", "club_games.csv", "countries.csv", "national_teams.csv"}


def sha256_stream(fh, chunk=1 << 20):
    h = hashlib.sha256()
    for block in iter(lambda: fh.read(chunk), b""):
        h.update(block)
    return h.hexdigest()


def current() -> dict:
    sums = {}
    with open(ZIP, "rb") as fh:
        sums[ZIP.name] = sha256_stream(fh)
    with zipfile.ZipFile(ZIP) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            with zf.open(info) as fh:
                sums[info.filename] = sha256_stream(fh)
    return sums


def main() -> int:
    if not ZIP.exists():
        print(f"missing {ZIP.relative_to(ROOT)} - download the dataset first (see README)")
        return 1
    sums = current()
    if "--write" in sys.argv:
        lines = ["# SHA-256 of the dataset version used in the paper (zip, then each CSV inside it).",
                 "# Files marked 'used' are read by the notebooks; the others are in the zip but not used."]
        for name, h in sums.items():
            tag = "zip" if name == ZIP.name else ("used" if name in USED else "not used")
            lines.append(f"{h}  {name}  # {tag}")
        SUMS.write_text("\n".join(lines) + "\n")
        print(f"wrote {SUMS.relative_to(ROOT)} ({len(sums)} files)")
        return 0
    expected = {}
    for line in SUMS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, name = line.split("#")[0].split()
            expected[name] = h
    bad = 0
    for name, h in expected.items():
        status = "OK      " if sums.get(name) == h else ("MISSING " if name not in sums else "DIFFERS ")
        if status != "OK      " and (name in USED or name == ZIP.name):
            bad += 1
        print(f"{status} {name}")
    print("same dataset version as the paper" if bad == 0 else
          f"{bad} used file(s) differ: results will not match the paper exactly")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
