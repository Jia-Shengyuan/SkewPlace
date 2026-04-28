import json
import logging
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSTALL = os.path.join(ROOT, "install")
INSTALL_DREAMPLACE = os.path.join(INSTALL, "dreamplace")
if INSTALL not in sys.path:
    sys.path.insert(0, INSTALL)
if INSTALL_DREAMPLACE not in sys.path:
    sys.path.insert(0, INSTALL_DREAMPLACE)

from dreamplace import NonLinearPlace
from dreamplace import Params
from dreamplace import PlaceDB
from dreamplace import Timer
from dreamplace.ops.timing.useful_skew import solve_useful_skew


def _abs_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _count_violations(edges, key):
    count = 0
    total_negative_slack = 0.0
    worst_slack = None
    for edge in edges:
        value = _finite_or_none(edge.get(key))
        if value is None:
            continue
        if worst_slack is None or value < worst_slack:
            worst_slack = value
        if value < 0.0:
            count += 1
            total_negative_slack += value
    return {
        "worst_slack": worst_slack,
        "tns": total_negative_slack,
        "violating_edges": count,
    }


def _clock_period_ps(sdc_path):
    pattern = re.compile(r"create_clock\b.*?-period\s+([0-9eE+\-.]+)")
    with open(sdc_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


def _skew_stats(skews):
    if not skews:
        return {
            "max_abs_skew": None,
            "skew_span": None,
        }
    values = list(skews.values())
    return {
        "max_abs_skew": max(abs(value) for value in values),
        "skew_span": max(values) - min(values),
    }


def run(config_path, checkpoint_path, baseline_n=100, max_skews=None, sweep_ns=None):
    params = Params.Params()
    params.load(config_path)

    params.lef_input = _abs_path(ROOT, params.lef_input)
    params.def_input = _abs_path(ROOT, params.def_input)
    params.verilog_input = _abs_path(ROOT, params.verilog_input)
    params.early_lib_input = _abs_path(ROOT, params.early_lib_input)
    params.late_lib_input = _abs_path(ROOT, params.late_lib_input)
    params.sdc_input = _abs_path(ROOT, params.sdc_input)
    params.result_dir = os.path.join(ROOT, "results")
    params.legalize_flag = 0
    params.detailed_place_flag = 0

    if max_skews is None:
        max_skews = [50.0, 100.0, 250.0, 500.0, 1000.0, None]
    if sweep_ns is None:
        sweep_ns = [1, 10, 100]

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer.load(params, placedb, checkpoint_path)

    timing_op = placer.op_collections.timing_op
    pos = placer.pos[0].data.clone().cpu()
    timing_op(pos)
    timing_op.timer.update_timing()

    baseline_graph = timing_op.timer.export_reg2reg_timing_graph(n=baseline_n, include_paths=False)
    baseline = {
        "n": baseline_n,
        "path_counts": baseline_graph.get("path_counts"),
        "num_registers": baseline_graph["num_registers"],
        "num_edges": baseline_graph["num_edges"],
        "setup": _count_violations(baseline_graph["edges"], "setup_slack"),
        "hold": _count_violations(baseline_graph["edges"], "hold_slack"),
    }

    sweeps = []
    for n in sweep_ns:
        graph = timing_op.timer.export_reg2reg_timing_graph(n=n, include_paths=False)
        for max_skew in max_skews:
            solution = solve_useful_skew(graph, max_skew=max_skew)
            skew_stats = _skew_stats(solution["skews"])
            sweeps.append({
                "n": n,
                "max_skew": max_skew,
                "path_counts": graph.get("path_counts"),
                "num_registers": graph["num_registers"],
                "num_edges": graph["num_edges"],
                "skew_margin": solution["margin"],
                "skew_success": solution["success"],
                "skew_status": solution["status"],
                "num_constraints": solution["num_constraints"],
                "max_abs_skew": skew_stats["max_abs_skew"],
                "skew_span": skew_stats["skew_span"],
                "delta_vs_setup_worst": None
                if baseline["setup"]["worst_slack"] is None or solution["margin"] is None
                else solution["margin"] - baseline["setup"]["worst_slack"],
            })

    summary = {
        "design": params.design_name(),
        "checkpoint_path": checkpoint_path,
        "clock_period_ps": _clock_period_ps(params.sdc_input),
        "baseline": baseline,
        "sweeps": sweeps,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python sweep_useful_skew_checkpoint.py <json> <checkpoint.pklz> [baseline_n] [max_skews] [sweep_ns]"
        )

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    baseline_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    max_skews = None
    if len(sys.argv) >= 5:
        max_skews = []
        for item in sys.argv[4].split(","):
            token = item.strip().lower()
            if not token:
                continue
            if token in ("none", "unbounded", "inf"):
                max_skews.append(None)
            else:
                max_skews.append(float(item))
    sweep_ns = [int(item) for item in sys.argv[5].split(",") if item] if len(sys.argv) >= 6 else None

    run(config_path, checkpoint_path, baseline_n=baseline_n, max_skews=max_skews, sweep_ns=sweep_ns)
