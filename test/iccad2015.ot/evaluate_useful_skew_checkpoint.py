import json
import logging
import math
import os
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


def _summarize_edges(edges, key_prefix, limit=10):
    rows = []
    slack_key = f"{key_prefix}_slack"
    delay_key = f"{key_prefix}_delay"
    constraint_key = f"{key_prefix}_constraint"
    filtered = [
        edge for edge in edges
        if _finite_or_none(edge.get(slack_key)) is not None
    ]
    filtered.sort(key=lambda edge: _finite_or_none(edge[slack_key]))
    for edge in filtered[:limit]:
        rows.append({
            "launch_register": edge["launch_register"],
            "capture_register": edge["capture_register"],
            slack_key: edge.get(slack_key),
            delay_key: edge.get(delay_key),
            constraint_key: edge.get(constraint_key),
        })
    return rows


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


def _sample_skews(skews, limit=10):
    return dict(list(sorted(skews.items()))[:limit])


def _baseline_metrics(timer, n):
    max_paths = timer.report_test_paths_by_split("max", n=n)
    min_paths = timer.report_test_paths_by_split("min", n=n)
    graph = timer.export_reg2reg_timing_graph(n=n, include_paths=False)
    return {
        "n": n,
        "path_counts": {"max": len(max_paths), "min": len(min_paths)},
        "num_registers": graph["num_registers"],
        "num_edges": graph["num_edges"],
        "setup": _count_violations(graph["edges"], "setup_slack"),
        "hold": _count_violations(graph["edges"], "hold_slack"),
        "worst_setup_edges": _summarize_edges(graph["edges"], "setup"),
        "worst_hold_edges": _summarize_edges(graph["edges"], "hold"),
    }


def _useful_skew_metrics(timer, n, max_skew):
    graph = timer.export_reg2reg_timing_graph(n=n, include_paths=False)
    skew = timer.solve_useful_skew(n=n, max_skew=max_skew)
    return {
        "n": n,
        "path_counts": graph.get("path_counts"),
        "num_registers": graph["num_registers"],
        "num_edges": graph["num_edges"],
        "skew_success": skew["success"],
        "skew_status": skew["status"],
        "skew_message": skew["message"],
        "skew_margin": skew["margin"],
        "num_constraints": skew["num_constraints"],
        "worst_setup_edges": _summarize_edges(graph["edges"], "setup"),
        "worst_hold_edges": _summarize_edges(graph["edges"], "hold"),
        "sample_skews": _sample_skews(skew["skews"]),
    }


def run(config_path, checkpoint_path, baseline_n=100, max_skew=50.0, sweep_ns=None):
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

    if sweep_ns is None:
        sweep_ns = [baseline_n]

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

    baseline = _baseline_metrics(timing_op.timer, baseline_n)
    useful_skew_sweep = [
        _useful_skew_metrics(timing_op.timer, item_n, max_skew)
        for item_n in sweep_ns
    ]

    summary = {
        "design": params.design_name(),
        "checkpoint_path": checkpoint_path,
        "baseline": baseline,
        "useful_skew_sweep": useful_skew_sweep,
        "max_skew": max_skew,
    }

    out_dir = os.path.join(ROOT, "results", "iccad2015.ot")
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_tag = os.path.basename(checkpoint_path).replace(".pklz", "")
    out_path = os.path.join(out_dir, checkpoint_tag + "_useful_skew.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(json.dumps({"summary_path": out_path}))


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python evaluate_useful_skew_checkpoint.py <json> <checkpoint.pklz> [baseline_n] [max_skew] [sweep_ns]"
        )

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    baseline_n = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    max_skew = float(sys.argv[4]) if len(sys.argv) >= 5 else 50.0
    sweep_ns = [int(item) for item in sys.argv[5].split(",") if item] if len(sys.argv) >= 6 else None

    run(config_path, checkpoint_path, baseline_n=baseline_n, max_skew=max_skew, sweep_ns=sweep_ns)
