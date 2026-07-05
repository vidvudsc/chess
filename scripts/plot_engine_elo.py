#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IdeaReport:
    idea: str
    label: str
    path: Path
    status: str
    baseline_elo: float = 0.0


REPORTS = [
    IdeaReport(
        idea="001",
        label="King safety + quiet checks",
        path=ROOT / "current" / "new_vs_old_idea001_repro_8g.json",
        status="rejected",
    ),
    IdeaReport(
        idea="002",
        label="TT 2^20",
        path=ROOT / "current" / "new_vs_old_idea002_tt20_60g.json",
        status="rejected",
    ),
    IdeaReport(
        idea="003",
        label="No quiet futility",
        path=ROOT / "current" / "new_vs_old_idea003_no_quiet_futility_60g.json",
        status="kept",
    ),
    IdeaReport(
        idea="003c",
        label="No quiet futility confirm",
        path=ROOT / "current" / "new_vs_old_idea003_no_quiet_futility_confirm_120g.json",
        status="kept",
    ),
    IdeaReport(
        idea="005",
        label="Wider qsearch delta",
        path=ROOT / "current" / "new_vs_idea003_idea005_qdelta_confirm_120g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="006",
        label="TT move depth gate",
        path=ROOT / "current" / "new_vs_idea003_idea006_ttmove_depth2_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="008",
        label="Early queen penalty",
        path=ROOT / "current" / "new_vs_idea003_idea008_queen_dev_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="009",
        label="Advanced pawn extend",
        path=ROOT / "current" / "new_vs_idea003_idea009_pawn_extend_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="010",
        label="History alpha malus",
        path=ROOT / "current" / "new_vs_idea003_idea010_history_alpha_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="011",
        label="Bishop/knight values",
        path=ROOT / "current" / "new_vs_idea003_idea011_bishop_knight_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="012",
        label="SEE capture order",
        path=ROOT / "current" / "new_vs_idea003_idea012_see_capture_order_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="013",
        label="No reverse futility",
        path=ROOT / "current" / "new_vs_idea003_idea013_no_reverse_futility_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="015",
        label="Rook on seventh",
        path=ROOT / "current" / "new_vs_idea003_idea015_rook7_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="016",
        label="King shield 28",
        path=ROOT / "current" / "new_vs_idea003_idea016_king_shield28_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="018",
        label="Repetition contempt",
        path=ROOT / "current" / "new_vs_idea003_idea018_repetition_contempt_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="019",
        label="Tempo 8",
        path=ROOT / "current" / "new_vs_idea003_idea019_tempo8_60g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="023",
        label="Lower bishop pair",
        path=ROOT / "current" / "new_vs_idea003_idea023_bishop_pair_low_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="024",
        label="Null move R+1",
        path=ROOT / "current" / "new_vs_idea003_idea024_nullmove_r3_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="025",
        label="Lower outpost",
        path=ROOT / "current" / "new_vs_idea003_idea025_outpost_low_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="026",
        label="Hanging penalty",
        path=ROOT / "current" / "new_vs_idea003_idea026_hanging_penalty_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="027",
        label="PV move order",
        path=ROOT / "current" / "new_vs_idea003_idea027_pv_order_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="029",
        label="TT 2^20 retest",
        path=ROOT / "current" / "new_vs_idea003_idea029_tt20_retest_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="030",
        label="TT lower score",
        path=ROOT / "current" / "new_vs_idea003_idea030_tt_lower_score_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="031",
        label="Positive SEE order",
        path=ROOT / "current" / "new_vs_idea003_idea031_see_positive_order_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="032",
        label="Qdelta 100",
        path=ROOT / "current" / "new_vs_idea003_idea032_qdelta100_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="035",
        label="Low root order",
        path=ROOT / "current" / "new_vs_idea003_idea035_root_bonus_low_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
    IdeaReport(
        idea="036",
        label="Back-rank flight",
        path=ROOT / "current" / "new_vs_idea003_idea036_backrank_flight_smoke_16g.json",
        status="rejected",
        baseline_elo=40.71957324001494,
    ),
]


def candidate_h2h(report_path: Path) -> Optional[dict]:
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    rows = data.get("head_to_head") or []
    for row in rows:
        if row.get("candidate") == "new":
            return row
    return rows[0] if rows else None


def standing_for_new(report_path: Path) -> Optional[dict]:
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    for row in data.get("standings", []):
        if row.get("name") == "new":
            return row
    return None


def main() -> int:
    points = []
    for idx, report in enumerate(REPORTS, start=1):
        h2h = candidate_h2h(report.path)
        standing = standing_for_new(report.path)
        if h2h is None or standing is None:
            continue
        points.append(
            {
                "x": idx,
                "idea": report.idea,
                "label": report.label,
                "status": report.status,
                "elo": report.baseline_elo + float(h2h.get("elo_diff") or 0.0),
                "incremental_elo": float(h2h.get("elo_diff") or 0.0),
                "ci_low": h2h.get("elo_ci_low"),
                "ci_high": h2h.get("elo_ci_high"),
                "games": int(h2h.get("games") or standing.get("games") or 0),
                "wins": int(standing.get("wins") or 0),
                "draws": int(standing.get("draws") or 0),
                "losses": int(standing.get("losses") or 0),
            }
        )

    if not points:
        raise SystemExit("no report data found")

    xs = [p["x"] for p in points]
    elos = [p["elo"] for p in points]
    best = []
    running = 0.0
    for elo in elos:
        running = max(running, elo)
        best.append(running)

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor("#f7f5ef")
    ax.set_facecolor("#fffdf8")

    for p in points:
        color = "#a9a9a9"
        edge = "#686868"
        z = 3
        if p["status"] == "rejected":
            color = "#d7b6a7"
            edge = "#9b5942"
        elif p["status"] == "kept":
            color = "#5ab769"
            edge = "#1f7a32"
            z = 5
        elif p["elo"] > 0:
            color = "#86b8d9"
            edge = "#2d6f9f"
        low = None if p["ci_low"] is None else p["ci_low"] + (p["elo"] - p["incremental_elo"])
        high = None if p["ci_high"] is None else p["ci_high"] + (p["elo"] - p["incremental_elo"])
        if low is not None and high is not None:
            yerr = [[p["elo"] - float(low)], [float(high) - p["elo"]]]
            ax.errorbar(
                p["x"],
                p["elo"],
                yerr=yerr,
                fmt="none",
                ecolor="#8d8a82",
                elinewidth=1.2,
                capsize=4,
                alpha=0.55,
                zorder=2,
            )
        ax.scatter(p["x"], p["elo"], s=92, color=color, edgecolor=edge, linewidth=1.3, zorder=z)
        ax.annotate(
            f"{p['elo']:+.1f} cum\n{p['wins']}W/{p['draws']}D/{p['losses']}L",
            (p["x"], p["elo"]),
            textcoords="offset points",
            xytext=(0, 13 if p["elo"] >= 0 else -38),
            ha="center",
            va="bottom" if p["elo"] >= 0 else "top",
            fontsize=8,
            color="#2f2e2b",
        )

    ax.plot(xs, best, color="#2f6f4e", linewidth=2.4, marker="o", markersize=4, zorder=4)
    ax.axhline(0, color="#38352f", linewidth=1.0, alpha=0.55)

    ax.set_title("HCE Engine Autoresearch: Elo Delta vs Ideas", fontsize=15, weight="bold", pad=16)
    ax.set_ylabel("Elo delta vs old local release")
    ax.set_xlabel("Idea")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p['idea']}\n{p['label']}" for p in points], fontsize=8)
    ax.grid(axis="y", color="#d8d2c5", linewidth=0.8, alpha=0.75)
    ax.grid(axis="x", visible=False)

    ax.text(
        0.012,
        0.965,
        "Line: best observed cumulative Elo. Points: individual ideas; CI bars use each test_lab match.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#55514a",
    )

    y_min = min(min(elos), min(best), 0.0) - 35.0
    y_max = max(max(elos), max(best), 0.0) + 45.0
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    out = ROOT / "current" / "engine_elo_by_idea.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
