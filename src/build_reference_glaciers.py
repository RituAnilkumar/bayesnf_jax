"""
src/build_reference_glaciers.py

Builds the pooled reference + former-reference + benchmark glacier metadata
table used for the revised (per-glacier WGMS) ensemble-selection validation
set, described in the "revised ensembling" discussion.

Join logic (deliberately strict — see CLAUDE.md discussion trail):
  1. Match each (country, name) entry from the hard-coded WGMS reference /
     former-reference / benchmark lists below to exactly one glacier_id in
     mass_balance.csv, by country + name-normalised equality (accent/
     punctuation-insensitive — WGMS uses ASCII transliteration, e.g.
     'STORGLACIÄREN' -> 'STORGLACIAEREN').
  2. If zero or more-than-one glacier_id matches -> DISCARD (ambiguous / not
     found). No fuzzy/partial matching is attempted; ambiguity is reported,
     not guessed at.
  3. Join the matched glacier_id onto glacier.csv's `id` column to fetch
     rgi60_ids. If missing/empty -> DISCARD (no RGI mapping available from
     these two files).
  4. Kept rows get an rgi_region parsed from rgi60_ids (e.g.
     'RGI60-10.01737' -> region 10).

Output: validation_data/per_gla/reference_benchmark_glaciers.csv
        (one row per input glacier, kept or discarded, with reasons)

This script only builds the glacier metadata/match table. It does NOT yet
build the per-year timeseries file or touch ensemble selection — those are
separate steps in the agreed plan.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

MASS_BALANCE_CSV = Path("validation_data/per_gla/mass_balance.csv")
GLACIER_CSV      = Path("validation_data/per_gla/glacier.csv")
OUTPUT_CSV       = Path("validation_data/per_gla/reference_benchmark_glaciers.csv")

# ---------------------------------------------------------------------------
# Input lists, as pasted by the user. (region label is WGMS mountain-range
# naming, kept only for provenance/display — NOT used for the validation/
# test split, since it doesn't align 1:1 with RGI regions.)
# ---------------------------------------------------------------------------

REFERENCE_GLACIERS = [
    ("ALASKA", "US", "GULKANA"),
    ("ALASKA", "US", "WOLVERINE"),
    ("WESTERN NORTH AMERICA", "US", "COLUMBIA (2057)"),
    ("WESTERN NORTH AMERICA", "US", "EASTON"),
    ("WESTERN NORTH AMERICA", "US", "LEMON CREEK"),
    ("WESTERN NORTH AMERICA", "US", "RAINBOW"),
    ("WESTERN NORTH AMERICA", "US", "SOUTH CASCADE"),
    ("WESTERN NORTH AMERICA", "CA", "HELM"),
    ("WESTERN NORTH AMERICA", "CA", "PEYTO"),
    ("WESTERN NORTH AMERICA", "CA", "PLACE"),
    ("ARCTIC CANADA NORTH & SOUTH", "CA", "DEVON ICE CAP NW"),
    ("ARCTIC CANADA NORTH & SOUTH", "CA", "MEIGHEN ICE CAP"),
    ("ARCTIC CANADA NORTH & SOUTH", "CA", "MELVILLE SOUTH ICE CAP"),
    ("ARCTIC CANADA NORTH & SOUTH", "CA", "WHITE"),
    ("ICELAND", "IS", "BRÚARJÖKULL"),
    ("ICELAND", "IS", "EYJABAKKAJÖKULL"),
    ("ICELAND", "IS", "HOFSJÖKULL E"),
    ("ICELAND", "IS", "HOFSJÖKULL N"),
    ("ICELAND", "IS", "HOFSJÖKULL SW"),
    ("ICELAND", "IS", "TUNGNÁRJÖKULL"),
    ("SVALBARD & JAN MAYEN", "SJ", "AUSTRE BROEGGERBREEN"),
    ("SVALBARD & JAN MAYEN", "SJ", "MIDTRE LOVÉNBREEN"),
    ("SCANDINAVIA", "NO", "AALFOTBREEN"),
    ("SCANDINAVIA", "NO", "ENGABREEN"),
    ("SCANDINAVIA", "NO", "GRAASUBREEN"),
    ("SCANDINAVIA", "NO", "HELLSTUGUBREEN"),
    ("SCANDINAVIA", "NO", "LANGFJORDJOEKELEN"),
    ("SCANDINAVIA", "NO", "NIGARDSBREEN"),
    ("SCANDINAVIA", "NO", "REMBESDALSKAAKA"),
    ("SCANDINAVIA", "NO", "STORBREEN"),
    ("SCANDINAVIA", "SE", "MARMAGLACIÄREN"),
    ("SCANDINAVIA", "SE", "RABOTS GLACIÄR"),
    ("SCANDINAVIA", "SE", "RIUKOJIETNA"),
    ("SCANDINAVIA", "SE", "STORGLACIÄREN"),
    ("CENTRAL EUROPE", "AT", "GOLDBERGKEES"),
    ("CENTRAL EUROPE", "AT", "HINTEREISFERNER"),
    ("CENTRAL EUROPE", "AT", "JAMTALFERNER"),
    ("CENTRAL EUROPE", "AT", "KESSELWANDFERNER"),
    ("CENTRAL EUROPE", "AT", "PASTERZE"),
    ("CENTRAL EUROPE", "AT", "VERNAGTFERNER"),
    ("CENTRAL EUROPE", "CH", "ALLALIN"),
    ("CENTRAL EUROPE", "CH", "BASÒDINO"),
    ("CENTRAL EUROPE", "CH", "CLARIDEN"),
    ("CENTRAL EUROPE", "CH", "GIÉTRO"),
    ("CENTRAL EUROPE", "CH", "GRIES"),
    ("CENTRAL EUROPE", "CH", "SILVRETTA"),
    ("CENTRAL EUROPE", "ES", "MALADETA"),
    ("CENTRAL EUROPE", "FR", "ARGENTIÈRE"),
    ("CENTRAL EUROPE", "FR", "SAINT SORLIN"),
    ("CENTRAL EUROPE", "IT", "CARESÈR"),
    ("CENTRAL EUROPE", "IT", "CIARDONEY"),
    ("CAUCASUS & MIDDLE EAST", "RU", "DJANKUAT"),
    ("CAUCASUS & MIDDLE EAST", "RU", "GARABASHI"),
    ("CAUCASUS & MIDDLE EAST", "RU", "LEVIY AKTRU"),
    ("CENTRAL ASIA", "KG", "ABRAMOV"),
    ("CENTRAL ASIA", "KG", "GOLUBIN"),
    ("CENTRAL ASIA", "KG", "KARA-BATKAK"),
    ("CENTRAL ASIA", "KZ", "TS. TUYUKSUYSKIY"),
    ("CENTRAL ASIA", "CN", "URUMQI GLACIER NO. 1"),
    ("SOUTHERN ANDES", "BO", "ZONGO"),
    ("SOUTHERN ANDES", "CL", "ECHAURREN NORTE"),
]

FORMER_REFERENCE_GLACIERS = [
    # (region, country, name, reason)
    ("ALPS", "AT", "STUBACHER SONNBLICKKEES", "no recent direct glaciological measurements"),
    ("ALPS", "AT", "WURTENKEES", "influence of artificial snow management"),
    ("ALPS", "FR", "SARENNES", "discontinued observations"),
    ("ASIA NORTH", "RU", "LEVIY AKTRU", "discontinued observations"),
    ("ASIA NORTH", "RU", "MALIY AKTRU", "discontinued observations"),
    ("ASIA NORTH", "RU", "VOVOPADNIY (NO. 125)", "discontinued observations"),
]

BENCHMARK_GLACIERS = [
    ("GREENLAND", "GL", "FREYA"),
    ("GREENLAND", "GL", "MITTIVAKKAT"),
    ("ICELAND", "IS", "LANGJÖKULL ICE CAP"),
    ("SVALBARD & JAN MAYEN", "SJ", "IRENEBREEN"),
    ("SVALBARD & JAN MAYEN", "SJ", "WALDEMARBREEN"),
    ("SVALBARD & JAN MAYEN", "SJ", "WERENSKIOLDBREEN"),
    ("CENTRAL EUROPE", "FR", "OSSOUE"),
    ("CENTRAL ASIA", "CN", "PARLUNG NO. 94"),
    ("ASIA SOUTH WEST & SOUTH EAST", "IN", "CHHOTA SHIGRI"),
    ("ASIA SOUTH WEST & SOUTH EAST", "NP", "MERA"),
    ("ASIA SOUTH WEST & SOUTH EAST", "NP", "POKALDE"),
    ("ASIA SOUTH WEST & SOUTH EAST", "NP", "RIKHA SAMBA"),
    ("ASIA SOUTH WEST & SOUTH EAST", "NP", "YALA"),
    ("SOUTHERN ANDES", "AR", "MARTIAL ESTE"),
    ("SOUTHERN ANDES", "BO", "CHARQUINI SUR"),
    ("SOUTHERN ANDES", "CL", "MOCHO CHOSHUENCO SE"),
    ("SOUTHERN ANDES", "CO", "CONEJERAS"),
    ("SOUTHERN ANDES", "EC", "ANTIZANA 15 ALPHA"),
    ("NEW ZEALAND", "NZ", "BREWSTER"),
    ("NEW ZEALAND", "NZ", "ROLLESTON"),
    ("ANTARCTICA & SUBANTARCTIC ISLANDS", "AQ", "BAHÍA DEL DIABLO"),
    ("ANTARCTICA & SUBANTARCTIC ISLANDS", "AQ", "HURD"),
    ("ANTARCTICA & SUBANTARCTIC ISLANDS", "AQ", "JOHNSONS"),
]


# ---------------------------------------------------------------------------
# Name normalisation (accent/punctuation-insensitive equality only —
# no fuzzy/partial matching, per the "discard if it doesn't map well" rule)
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Strip accents, uppercase, collapse punctuation/whitespace.

    'STORGLACIÄREN' -> 'STORGLACIAEREN' (WGMS ASCII transliteration is not
    a simple accent-strip: Ä/Ö/Å become AE/OE/AA, not A/O/A). We therefore
    try both a plain accent-strip AND the transliterated form when matching.
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re.sub(r"[.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _transliterated(name: str) -> str:
    """Apply the Scandinavian/Icelandic ASCII transliteration WGMS uses
    (Ä->AE, Ö->OE, Å->AA) before the plain accent-strip normalisation."""
    repl = {
        "Ä": "AE", "ä": "ae",
        "Ö": "OE", "ö": "oe",
        "Å": "AA", "å": "aa",
        "Ü": "UE", "ü": "ue",
    }
    for k, v in repl.items():
        name = name.replace(k, v)
    return _normalise(name)


# Manual overrides: (country, input_name) -> exact WGMS glacier_name, for
# confirmed cases where WGMS uses a documented abbreviation/spelling that no
# generic normalisation rule should guess at (verified by hand against
# mass_balance.csv before adding — see conversation trail). Not a fuzzy-match
# fallback: these are exact, hand-checked substitutions only.
NAME_OVERRIDES = {
    ("AT", "GOLDBERGKEES"): "GOLDBERG K.",
    ("AT", "HINTEREISFERNER"): "HINTEREIS F.",
    ("AT", "JAMTALFERNER"): "JAMTAL F.",
    ("AT", "KESSELWANDFERNER"): "KESSELWAND F.",
    ("AT", "VERNAGTFERNER"): "VERNAGT F.",
    ("AT", "STUBACHER SONNBLICKKEES"): "STUBACHER SONNBLICK K.",
    ("AT", "WURTENKEES"): "OE. WURTEN K.",
    ("CH", "CLARIDEN"): "CLARIDENFIRN",
    ("EC", "ANTIZANA 15 ALPHA"): "ANTIZANA15ALPHA",
}


def build_norm_lookup(mb: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    """(country, normalised_name) -> list of glacier_id candidates."""
    lookup: dict[tuple[str, str], list[int]] = {}
    sub = mb[["country", "glacier_name", "glacier_id"]].drop_duplicates()
    for _, row in sub.iterrows():
        country = str(row["country"]).strip().upper()
        for norm in {_normalise(row["glacier_name"]), _transliterated(row["glacier_name"])}:
            lookup.setdefault((country, norm), [])
            if row["glacier_id"] not in lookup[(country, norm)]:
                lookup[(country, norm)].append(row["glacier_id"])
    return lookup


def match_one(country: str, name: str, lookup: dict[tuple[str, str], list[int]]) -> tuple[list[int], str]:
    """Return (candidate glacier_ids, which normalisation matched)."""
    country = country.strip().upper()

    override = NAME_OVERRIDES.get((country, name.strip().upper()))
    if override is not None:
        norm = _normalise(override)
        cands = lookup.get((country, norm), [])
        if cands:
            return cands, "manual_override"

    for norm_fn, label in [(_normalise, "plain"), (_transliterated, "transliterated")]:
        norm = norm_fn(name)
        cands = lookup.get((country, norm), [])
        if cands:
            return cands, label
    return [], "none"


def _rgi_region(rgi_id: str) -> int | None:
    m = re.match(r"RGI\d+-(\d+)\.", str(rgi_id))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build() -> pd.DataFrame:
    mb = pd.read_csv(MASS_BALANCE_CSV, low_memory=False)
    gl = pd.read_csv(GLACIER_CSV, low_memory=False)
    gl_by_id = gl.set_index("id")

    lookup = build_norm_lookup(mb)

    all_entries = []
    for region, country, name in REFERENCE_GLACIERS:
        all_entries.append((region, country, name, "reference", ""))
    for region, country, name, reason in FORMER_REFERENCE_GLACIERS:
        all_entries.append((region, country, name, "former_reference", reason))
    for region, country, name in BENCHMARK_GLACIERS:
        all_entries.append((region, country, name, "benchmark", ""))

    rows = []
    for region, country, name, source, reason in all_entries:
        cands, match_type = match_one(country, name, lookup)

        status = "kept"
        discard_reason = ""
        glacier_id = None
        matched_wgms_name = ""
        rgi_id = ""
        rgi_region = None

        if len(cands) == 0:
            status = "discarded"
            discard_reason = "no glacier_id match in mass_balance.csv"
        elif len(cands) > 1:
            status = "discarded"
            discard_reason = f"ambiguous: {len(cands)} glacier_id candidates {cands}"
        else:
            glacier_id = int(cands[0])
            matched_rows = mb[mb["glacier_id"] == glacier_id]
            matched_wgms_name = matched_rows["glacier_name"].iloc[0]

            if glacier_id not in gl_by_id.index:
                status = "discarded"
                discard_reason = f"glacier_id {glacier_id} not found in glacier.csv"
            else:
                rgi_raw = gl_by_id.loc[glacier_id, "rgi60_ids"]
                if pd.isna(rgi_raw) or not str(rgi_raw).strip():
                    status = "discarded"
                    discard_reason = f"no rgi60_ids for glacier_id {glacier_id} in glacier.csv"
                elif "|" in str(rgi_raw):
                    status = "discarded"
                    discard_reason = f"multiple rgi60_ids for glacier_id {glacier_id}: {rgi_raw}"
                else:
                    rgi_id = str(rgi_raw).strip()
                    rgi_region = _rgi_region(rgi_id)

        rows.append({
            "input_region_label": region,
            "country": country,
            "input_name": name,
            "source_list": source,
            "former_reference_reason": reason,
            "glacier_id": glacier_id,
            "matched_wgms_name": matched_wgms_name,
            "match_type": match_type,
            "rgi_id": rgi_id,
            "rgi_region": rgi_region,
            "status": status,
            "discard_reason": discard_reason,
        })

    return _dedupe_by_glacier_id(pd.DataFrame(rows))


def _dedupe_by_glacier_id(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that resolve to the same glacier_id (e.g. a glacier
    listed both as 'reference' and 'former_reference' with different labels).

    Priority for which row's fields to keep: reference > benchmark >
    former_reference. Other source lists it also appeared under, and any
    former_reference_reason, are preserved in `also_listed_as` / `notes`.
    """
    kept_mask = df["status"] == "kept"
    kept = df[kept_mask].copy()
    other = df[~kept_mask].copy()

    priority = {"reference": 0, "benchmark": 1, "former_reference": 2}
    kept["_prio"] = kept["source_list"].map(priority)

    merged_rows = []
    for glacier_id, group in kept.groupby("glacier_id", sort=False):
        if len(group) == 1:
            row = group.iloc[0].to_dict()
            row["also_listed_as"] = ""
            row["notes"] = ""
        else:
            group = group.sort_values("_prio")
            primary = group.iloc[0].to_dict()
            other_sources = group.iloc[1:]
            primary["also_listed_as"] = "|".join(sorted(set(other_sources["source_list"])))
            reasons = [r for r in other_sources["former_reference_reason"] if r]
            primary["notes"] = (
                f"duplicate across input lists (glacier_id={glacier_id}); "
                f"other reason(s): {'; '.join(reasons)}" if reasons else
                f"duplicate across input lists (glacier_id={glacier_id})"
            )
            row = primary
        row.pop("_prio", None)
        merged_rows.append(row)

    kept_deduped = pd.DataFrame(merged_rows)
    other["also_listed_as"] = ""
    other["notes"] = ""
    return pd.concat([kept_deduped, other], ignore_index=True, sort=False)


def main() -> None:
    df = build()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    n_total = len(df)
    kept = df[df["status"] == "kept"]
    discarded = df[df["status"] == "discarded"]

    print(f"Total input glaciers : {n_total}")
    print(f"Kept                 : {len(kept)}")
    print(f"Discarded            : {len(discarded)}")
    print(f"\nSaved -> {OUTPUT_CSV}")

    if not discarded.empty:
        print("\n=== Discarded glaciers ===")
        for _, r in discarded.iterrows():
            print(f"  [{r['source_list']:17s}] {r['country']} {r['input_name']:30s} "
                  f"-> {r['discard_reason']}")

    if not kept.empty:
        dup_regions = kept["rgi_region"].value_counts().sort_index()
        print("\n=== Kept glaciers per RGI region ===")
        for region, n in dup_regions.items():
            print(f"  r{int(region):02d}: {n}")


if __name__ == "__main__":
    main()
