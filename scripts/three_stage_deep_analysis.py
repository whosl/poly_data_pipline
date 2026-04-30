#!/usr/bin/env python3
"""Deep analysis: top-893 comparison, overlap, p_first buckets."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly.training.features import infer_feature_columns

PARTS_DIR = Path("artifacts/training_rf_et_win/alpha_dataset_parts")
SP = 0.04


def main():
    # Discover features
    for date_dir in sorted(PARTS_DIR.iterdir()):
        if date_dir.is_dir() and date_dir.name.startswith("date="):
            files = sorted(date_dir.glob("*.parquet"))
            if files:
                df0 = pl.read_parquet(str(files[0]), n_rows=2000)
                feature_cols = infer_feature_columns(df0)
                break

    # Load models
    models = {}
    for name in ["s1_xgb", "s2_lgb", "s2_xgb", "s2_rf", "s2_et",
                 "s3_lgb", "s3_xgb", "s3_rf", "s3_et"]:
        subdir = "fill_models" if name.startswith("s2_") or name.startswith("s1_") else "unwind_models"
        models[name] = joblib.load(
            f"artifacts/three_stage_eval/{subdir}/{name}.joblib"
        )["model"]

    # Streaming predict test
    d = PARTS_DIR / "date=20260424"
    files = sorted(d.glob("*.parquet"))
    label_cols = ["y_first_leg_fill", "y_two_leg_entry_binary_10s",
                  "first_unwind_profit_proxy_10s", "recv_ns"]
    need = list(set(feature_cols + label_cols))

    def to_numpy(piece, fcols):
        arrs = []
        for c in fcols:
            if c not in piece.columns:
                arrs.append(np.full(piece.height, np.nan, np.float32))
            else:
                s = piece[c]
                if s.dtype in (pl.String, pl.Categorical):
                    s = s.cast(pl.Categorical).to_physical()
                arrs.append(s.to_numpy().astype(np.float32))
        return np.column_stack(arrs)

    all_preds = {k: [] for k in models}
    all_yc, all_yr, all_yf, all_recv = [], [], [], []

    for f in files:
        avail = set(pl.read_parquet_schema(str(f)).names())
        cols = [c for c in need if c in avail]
        piece = pl.read_parquet(str(f), columns=cols)
        if piece.is_empty():
            continue
        X = to_numpy(piece, feature_cols)
        for name, m in models.items():
            if "s3_" in name:
                all_preds[name].append(m.predict(X))
            else:
                all_preds[name].append(m.predict_proba(X)[:, 1])
        all_yc.append(piece["y_two_leg_entry_binary_10s"].to_numpy().astype(np.float32))
        all_yr.append(piece["first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32))
        yf = (
            piece["y_first_leg_fill"].to_numpy().astype(np.float32)
            if "y_first_leg_fill" in piece.columns
            else np.full(piece.height, np.nan, np.float32)
        )
        all_yf.append(np.nan_to_num(yf, nan=0.0))
        all_recv.append(piece["recv_ns"].to_numpy().astype(np.int64))
        del X, piece
        gc.collect()

    preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    y_cls = np.concatenate(all_yc)
    y_reg = np.concatenate(all_yr)
    y_first = np.concatenate(all_yf)
    recv = np.concatenate(all_recv)
    del all_preds, all_yc, all_yr, all_yf, all_recv
    gc.collect()

    # Embargo
    purge_ns = 300 * 1_000_000_000
    mask = recv >= recv.min() + purge_ns
    preds = {k: v[mask] for k, v in preds.items()}
    y_cls, y_reg, y_first = y_cls[mask], y_reg[mask], y_first[mask]
    N = len(y_cls)
    print(f"Test rows: {N:,}")

    # Configs to compare
    combos = [
        ("s1_xgb", "s2_rf", "s3_rf", 0.5, 0.75, 0.05, -0.05),
        ("s1_xgb", "s2_et", "s3_et", 0.5, 0.70, 0.05, -0.05),
    ]

    for ci, (s1_name, s2_name, s3_name, mpf1, mpf2, th, mu) in enumerate(combos):
        p1 = preds[s1_name]
        p2 = preds[s2_name]
        p3 = preds[s3_name]
        ep = p2 * SP + (1 - p2) * p3
        sel3 = (p1 >= mpf1) & (p2 >= mpf2) & (ep >= th) & (p3 >= mu)
        idx3 = np.where(sel3)[0]
        n3 = len(idx3)
        combo_label = f"{s1_name}+{s2_name}+{s3_name}"
        print(f"\n{'#' * 70}")
        print(f"COMBO {ci+1}: {combo_label}  cfg: p1>={mpf1} p2>={mpf2} thr>={th} u>={mu}")
        print(f"{'#' * 70}")
        print(f"3-stage selected: {n3}")

        # ==================================================================
        # ANALYSIS 1: True top-N by expected profit
        # ==================================================================
        print("\n  ANALYSIS 1: True top-N by expected profit")

        def report(name, idx):
            n = len(idx)
            if n == 0:
                return f"  {name:40s} n=     0  (empty)"
            ap = float(np.nanmean(y_reg[idx]))
            med = float(np.nanmedian(y_reg[idx]))
            wr = float(np.nanmean(y_cls[idx]))
            fr = float(np.nanmean(y_first[idx]))
            tot = float(np.nansum(y_reg[idx]))
            return (f"  {name:40s} n={n:5d}  avgP={ap:+.4f}  medP={med:+.4f}"
                    f"  wr={wr:.3f}  fill_rate={fr:.3f}  totalP={tot:+.1f}")

        print(report("3-stage (threshold)", idx3))

        # 2-stage top-N using same s2+s3
        top2s_idx = np.argsort(-ep)[:n3]
        print(report(f"2-stage top-{n3} by ep({s2_name}+{s3_name})", top2s_idx))

        # 2-stage best across all s2+s3 combos
        best_2s_name = ""
        best_2s_avg = -999
        best_2s_idx = None
        for s2n in ["s2_lgb", "s2_xgb", "s2_rf", "s2_et"]:
            for s3n in ["s3_lgb", "s3_xgb", "s3_rf", "s3_et"]:
                pp2 = preds[s2n]
                pp3 = preds[s3n]
                epp = pp2 * SP + (1 - pp2) * pp3
                top_idx = np.argsort(-epp)[:n3]
                a = float(np.nanmean(y_reg[top_idx]))
                if a > best_2s_avg:
                    best_2s_avg = a
                    best_2s_name = f"{s2n}+{s3n}"
                    best_2s_idx = top_idx

        print(report(f"2-stage BEST top-{n3} ({best_2s_name})", best_2s_idx))

        # ==================================================================
        # ANALYSIS 2: Overlap
        # ==================================================================
        print("\n  --- Overlap ---")

        def overlap_report(label, set_a, set_b, label_a, label_b):
            common = set_a & set_b
            only_a = set_a - set_b
            only_b = set_b - set_a
            ca = list(common) if common else []
            cb = list(only_a) if only_a else []
            cc = list(only_b) if only_b else []
            c_ap = float(np.nanmean(y_reg[ca])) if ca else float('nan')
            c_wr = float(np.nanmean(y_cls[ca])) if ca else float('nan')
            a_ap = float(np.nanmean(y_reg[cb])) if cb else float('nan')
            a_wr = float(np.nanmean(y_cls[cb])) if cb else float('nan')
            b_ap = float(np.nanmean(y_reg[cc])) if cc else float('nan')
            b_wr = float(np.nanmean(y_cls[cc])) if cc else float('nan')
            pct = len(common)/len(set_a) if len(set_a) > 0 else 0
            print(f"    vs {label}:")
            print(f"      Common:       {len(common):5d}/{len(set_a)} ({pct:.1%})  avgP={c_ap:+.4f}  wr={c_wr:.3f}")
            print(f"      {label_a:13s}: {len(only_a):5d}  avgP={a_ap:+.4f}  wr={a_wr:.3f}")
            print(f"      {label_b:13s}: {len(only_b):5d}  avgP={b_ap:+.4f}  wr={b_wr:.3f}")

        set3 = set(idx3.tolist())
        set2 = set(top2s_idx.tolist())
        overlap_report(f"2s top-{n3} ({s2_name}+{s3_name})", set3, set2, "3-stage only", "2-stage only")

        set2b = set(best_2s_idx.tolist())
        overlap_report(f"2s BEST top-{n3} ({best_2s_name})", set3, set2b, "3-stage only", "2-stage only")

        # ==================================================================
        # ANALYSIS 3: Bucket by p_first
        # ==================================================================
        print("\n  --- Bucket by p_first on 3-stage selected ---")
        buckets = [
            (0.0, 0.90, "< 0.90"),
            (0.90, 0.93, "0.90-0.93"),
            (0.93, 0.95, "0.93-0.95"),
            (0.95, 0.97, "0.95-0.97"),
            (0.97, 0.99, "0.97-0.99"),
            (0.99, 1.01, "0.99+"),
        ]
        hdr2 = f"      {'bucket':>10s} {'count':>8s} {'fill_rt':>8s} {'avgP':>8s} {'medP':>8s} {'wr':>6s}"
        print(hdr2)
        print("      " + "-" * (len(hdr2) - 6))
        for lo, hi, label in buckets:
            bmask = (p1[idx3] >= lo) & (p1[idx3] < hi)
            nb = bmask.sum()
            if nb < 3:
                print(f"      {label:>10s} {nb:8d}      -")
                continue
            bidx = idx3[bmask]
            fill_rt = float(y_first[bidx].mean())
            avg_p = float(np.nanmean(y_reg[bidx]))
            med_p = float(np.nanmedian(y_reg[bidx]))
            wr = float(np.nanmean(y_cls[bidx]))
            print(f"      {label:>10s} {nb:8d} {fill_rt:8.3f} {avg_p:+8.4f} {med_p:+8.4f} {wr:6.3f}")


if __name__ == "__main__":
    main()
