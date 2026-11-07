"""Download the three public battery aging datasets used in this study into data/raw/.

NASA PCoE Li-ion Battery Aging Datasets (B. Saha and K. Goebel, NASA Ames).
Oxford Battery Degradation Dataset 1 (C. Birkl and D. Howey), Open Database License.
CALCE CS2 cells 33-38 (Center for Advanced Life Cycle Engineering, Univ. of Maryland).
"""
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

SOURCES = {
    "nasa/nasa_battery.zip": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
    "oxford/Oxford_Battery_Degradation_Dataset_1.mat":
        "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/files/m5ac36a1e2073852e4f1f7dee647909a7",
    **{f"calce/CS2_{c}.zip": f"https://web.calce.umd.edu/batteries/data/CS2_{c}.zip" for c in range(33, 39)},
}


def fetch(rel, url):
    out = RAW / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {rel}")
        return out
    print(f"[get ] {rel}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.rename(out)
    return out


def unzip_all(path, dest):
    """Extract a zip, including any zip files nested inside it."""
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)
    for inner in list(Path(dest).rglob("*.zip")):
        if inner.resolve() != Path(path).resolve():
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.with_suffix(""))
            inner.unlink()


def main():
    for rel, url in SOURCES.items():
        p = fetch(rel, url)
        if p.suffix == ".zip":
            dest = p.with_suffix("")
            if not dest.exists():
                unzip_all(p, dest)


if __name__ == "__main__":
    main()
