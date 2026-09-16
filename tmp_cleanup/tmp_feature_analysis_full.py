"""Feature importance analysis — FULL (unmasked) features, all 19 regions."""
import pandas as pd
import numpy as np
import os

EXPLAIN_ROOT = "outputs/egu_fin/explain_te"
GROUP = "no_time_encoding"
FILE  = "attributions_finetune_ensemble1_no_time_encoding.csv"

ALL_FEATURES = [
    "year",
    "CenLat", "CenLon",
    "log_Area", "Zmed", "Slope", "Aspect", "Zmin", "Zmax",
    "t2m_abl_mean", "t2m_abl_std", "t2m_acc_mean", "t2m_acc_std",
    "tp_abl_sum", "tp_acc_sum",
    "ssrd_abl_sum", "ssrd_acc_sum",
]

CLIMATE_FEATS = {"t2m_abl_mean","t2m_abl_std","t2m_acc_mean","t2m_acc_std",
                 "tp_abl_sum","tp_acc_sum","ssrd_abl_sum","ssrd_acc_sum"}
GEOM_FEATS    = {"log_Area","Zmed","Slope","Aspect","Zmin","Zmax"}
COORD_FEATS   = {"CenLat","CenLon"}
YEAR_FEATS    = {"year"}

REGION_NAMES = {
    "r01":"Alaska","r02":"Western Canada & US","r03":"Arctic Canada North",
    "r04":"Arctic Canada South","r05":"Greenland Periphery","r06":"Iceland",
    "r07":"Svalbard","r08":"Scandinavia","r09":"Russian Arctic","r10":"North Asia",
    "r11":"Central Europe","r12":"Caucasus & Middle East","r13":"Central Asia",
    "r14":"South Asia West","r15":"South Asia East","r16":"Low Latitudes",
    "r17":"Southern Andes","r18":"New Zealand","r19":"Antarctic & Subantarctic",
}

results = {}

for rkey in [f"r{i:02d}" for i in range(1, 20)]:
    path = os.path.join(EXPLAIN_ROOT, rkey, GROUP, FILE)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)

    mean_abs, mean_signed = {}, {}
    for ft in ALL_FEATURES:
        col = f"attr_{ft}"
        if col in df.columns:
            mean_abs[ft]    = float(df[col].abs().mean())
            mean_signed[ft] = float(df[col].mean())
        else:
            mean_abs[ft] = mean_signed[ft] = 0.0

    total = sum(mean_abs.values()) or 1.0

    ranked = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)

    clim_pct  = 100*sum(mean_abs[f] for f in CLIMATE_FEATS)/total
    geom_pct  = 100*sum(mean_abs[f] for f in GEOM_FEATS)/total
    coord_pct = 100*sum(mean_abs[f] for f in COORD_FEATS)/total
    year_pct  = 100*sum(mean_abs[f] for f in YEAR_FEATS)/total

    # CenLat / CenLon direction
    lat_sign = "+" if mean_signed["CenLat"] > 0 else "-"
    lon_sign = "+" if mean_signed["CenLon"] > 0 else "-"
    lat_pct  = 100*mean_abs["CenLat"]/total
    lon_pct  = 100*mean_abs["CenLon"]/total

    results[rkey] = dict(name=REGION_NAMES[rkey], mean_abs=mean_abs,
                         mean_signed=mean_signed, ranked=ranked, total=total,
                         clim_pct=clim_pct, geom_pct=geom_pct,
                         coord_pct=coord_pct, year_pct=year_pct,
                         lat_pct=lat_pct, lon_pct=lon_pct,
                         lat_sign=lat_sign, lon_sign=lon_sign)

print("=" * 80)
print("FEATURE IMPORTANCE — FULL (unmasked), NO TIME ENCODING")
print("=" * 80)

for rkey in [f"r{i:02d}" for i in range(1, 20)]:
    if rkey not in results: continue
    r = results[rkey]
    print(f"\n{'─'*72}")
    print(f"  {rkey.upper()}  {r['name']}")
    print(f"{'─'*72}")
    print(f"  Group split:  Climate {r['clim_pct']:.0f}%  |  Geometry {r['geom_pct']:.0f}%  |  Coordinates {r['coord_pct']:.0f}%  |  Year {r['year_pct']:.0f}%")
    print(f"  CenLat:  mean_abs={r['mean_abs']['CenLat']:.4f} ({r['lat_pct']:.1f}%)  signed_mean={r['mean_signed']['CenLat']:+.4f}  (direction {r['lat_sign']})")
    print(f"  CenLon:  mean_abs={r['mean_abs']['CenLon']:.4f} ({r['lon_pct']:.1f}%)  signed_mean={r['mean_signed']['CenLon']:+.4f}  (direction {r['lon_sign']})")
    print(f"  Top 10 features (mean |attribution|, % of total):")
    for i, (ft, val) in enumerate(r["ranked"][:10], 1):
        pct  = 100*val/r["total"]
        sgn  = r["mean_signed"][ft]
        grp  = "clim" if ft in CLIMATE_FEATS else ("geom" if ft in GEOM_FEATS else ("coord" if ft in COORD_FEATS else "year"))
        print(f"    {i:2d}. {ft:<22s}  {val:.4f}  ({pct:.1f}%)  signed={sgn:+.4f}  [{grp}]")

print("\n\n" + "=" * 80)
print("SLIDE BULLET POINTS — UNMASKED (with CenLat / CenLon)")
print("=" * 80)

for rkey in [f"r{i:02d}" for i in range(1, 20)]:
    if rkey not in results: continue
    r = results[rkey]
    ranked = r["ranked"]
    top3   = [ft for ft, _ in ranked[:3]]

    clim  = r["clim_pct"]
    geom  = r["geom_pct"]
    coord = r["coord_pct"]
    lat_p = r["lat_pct"]
    lon_p = r["lon_pct"]
    lat_s = r["mean_signed"]["CenLat"]
    lon_s = r["mean_signed"]["CenLon"]

    # Interpret CenLat direction
    if lat_s > 0.05:
        lat_interp = f"positive (higher latitude → more negative MB on average within region)"
    elif lat_s < -0.05:
        lat_interp = f"negative (lower latitude → more negative MB — warmer southern glaciers lose more mass)"
    else:
        lat_interp = f"near-zero (no clear latitudinal gradient within region)"

    if lon_s > 0.05:
        lon_interp = f"positive (more eastern/continental position → different MB regime)"
    elif lon_s < -0.05:
        lon_interp = f"negative (more western/maritime position → different MB regime)"
    else:
        lon_interp = f"near-zero (no strong east-west gradient within region)"

    print(f"\n{'─'*62}")
    print(f"  {rkey.upper()} — {r['name']}")
    print(f"{'─'*62}")
    print(f"  • Climate {clim:.0f}%  /  Geometry {geom:.0f}%  /  Coordinates {coord:.0f}%  /  Year 0%")
    print(f"  • Top 3: {', '.join(top3)}")
    print(f"  • CenLat ({lat_p:.1f}%): {lat_interp}")
    print(f"  • CenLon ({lon_p:.1f}%): {lon_interp}")
    if coord > 10:
        print(f"  • Geographic position explains {coord:.0f}% — significant latitudinal/longitudinal structuring within region")
    elif coord > 5:
        print(f"  • Geographic position explains {coord:.0f}% — moderate spatial gradient within region")
    else:
        print(f"  • Geographic position explains only {coord:.0f}% — region is relatively homogeneous spatially")
