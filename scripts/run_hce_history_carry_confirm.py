#!/usr/bin/env python3
"""Resume confirmation of the carry100 history candidate after suite repair."""

from __future__ import annotations

import json
import time

try:
    from run_hce_history_carry_sweep import (
        SWEEP,
        VARIANTS,
        probability,
        run_hce_suite,
        run_match,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.run_hce_history_carry_sweep import (
        SWEEP,
        VARIANTS,
        probability,
        run_hce_suite,
        run_match,
        write_json,
    )


def main() -> int:
    status_path = SWEEP / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    name = "carry100"
    options = VARIANTS[name]
    status["status"] = "confirming_after_suite_reset_fix"
    status.pop("error", None)
    write_json(status_path, status)

    try:
        run_hce_suite(name, options)
        status["hce_suite_after_reset_fix"] = "passed"
        write_json(status_path, status)

        result_120 = run_match(
            f"gate_{name}_120g",
            options,
            games=120,
            positions=60,
            seed=20261090,
        )
        status.setdefault("gates_120", {})[name] = result_120
        write_json(status_path, status)

        advances = (
            result_120["engine_failures"] == 0
            and result_120["elo_diff"] is not None
            and result_120["elo_diff"] > 0
            and probability(result_120) >= 0.75
        )
        if not advances:
            status["status"] = "rejected_at_120"
            status["promotion"] = "none"
            status["finished_at_unix"] = int(time.time())
            write_json(status_path, status)
            return 0

        result_240 = run_match(
            f"confirm_{name}_240g",
            options,
            games=240,
            positions=120,
            seed=20261100,
        )
        status["confirmation_240"] = {
            "candidate": name,
            **result_240,
        }
        confirmed = (
            result_240["engine_failures"] == 0
            and result_240["elo_diff"] is not None
            and result_240["elo_diff"] > 0
            and probability(result_240) >= 0.90
        )
        status["status"] = "confirmed_candidate" if confirmed else "rejected_at_240"
        status["promotion"] = "blocked_pending_review" if confirmed else "none"
        status["finished_at_unix"] = int(time.time())
        write_json(status_path, status)
        return 0
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        status["promotion"] = "none"
        status["finished_at_unix"] = int(time.time())
        write_json(status_path, status)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
