import json
import logging
import math
import os
import sys

import numpy as np

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


def _sample_skews(skews, limit=10):
    return dict(list(sorted(skews.items()))[:limit])


def _scalar_to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _final_placement_metrics(metrics):
    if not metrics:
        return {}
    last = metrics[-1]
    return {
        "objective": _scalar_to_float(getattr(last, "objective", None)) if getattr(last, "objective", None) is not None else None,
        "hpwl": _scalar_to_float(getattr(last, "hpwl", None)) if getattr(last, "hpwl", None) is not None else None,
        "overflow": _scalar_to_float(getattr(last, "overflow", None)) if getattr(last, "overflow", None) is not None else None,
        "max_density": _scalar_to_float(getattr(last, "max_density", None)) if getattr(last, "max_density", None) is not None else None,
    }


def _baseline_metrics(timer, n):
    max_paths = timer.report_test_paths_by_split("max", n=n)
    min_paths = timer.report_test_paths_by_split("min", n=n)
    graph = timer.export_reg2reg_timing_graph(n=n, include_paths=False)
    setup_metrics = _count_violations(graph["edges"], "setup_slack")
    hold_metrics = _count_violations(graph["edges"], "hold_slack")
    return {
        "n": n,
        "path_counts": {
            "max": len(max_paths),
            "min": len(min_paths),
        },
        "num_registers": graph["num_registers"],
        "num_edges": graph["num_edges"],
        "setup": setup_metrics,
        "hold": hold_metrics,
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


def run(config_path, n=100, iterations=10, max_skew=50.0, sweep_ns=None):
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
    params.global_place_stages[0]["iteration"] = iterations

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    metrics = placer(params, placedb, learning_rate_value)

    timing_op = placer.op_collections.timing_op
    pos = placer.pos[0].data.clone().cpu()
    timing_op(pos)
    timing_op.timer.update_timing()

    if sweep_ns is None:
        sweep_ns = [n]

    baseline = _baseline_metrics(timing_op.timer, n=n)
    useful_skew_sweep = [
        _useful_skew_metrics(timing_op.timer, n=item_n, max_skew=max_skew)
        for item_n in sweep_ns
    ]

    summary = {
        "design": params.design_name(),
        "iterations": iterations,
        "n": n,
        "sweep_ns": sweep_ns,
        "max_skew": max_skew,
        "timer_engine": timer.timer_engine,
        "num_tests": timer.num_tests(),
        "placement_metrics": _final_placement_metrics(metrics),
        "report_template": {
            "baseline": [
                "placement_metrics.objective",
                "placement_metrics.hpwl",
                "placement_metrics.overflow",
                "placement_metrics.max_density",
                "path_counts.max",
                "path_counts.min",
                "num_registers",
                "num_edges",
                "setup.worst_slack",
                "setup.tns",
                "setup.violating_edges",
                "hold.worst_slack",
                "hold.tns",
                "hold.violating_edges",
            ],
            "useful_skew": [
                "n",
                "path_counts.max",
                "path_counts.min",
                "num_registers",
                "num_edges",
                "skew_success",
                "skew_margin",
                "num_constraints",
                "worst_setup_edges[0]",
                "worst_hold_edges[0]",
            ],
        },
        "baseline": baseline,
        "useful_skew_sweep": useful_skew_sweep,
    }

    out_dir = os.path.join(ROOT, "results", "iccad2015.ot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"{params.design_name()}_useful_skew_n{n}_iter{iterations}.json",
    )
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

    config_path = os.path.join(ROOT, "test", "iccad2015.ot", "superblue1.json")
    n = 100
    iterations = 10
    max_skew = 50.0
    sweep_ns = None

    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    if len(sys.argv) >= 3:
        n = int(sys.argv[2])
    if len(sys.argv) >= 4:
        iterations = int(sys.argv[3])
    if len(sys.argv) >= 5:
        max_skew = float(sys.argv[4])
    if len(sys.argv) >= 6:
        sweep_ns = [int(item) for item in sys.argv[5].split(",") if item]

    run(config_path, n=n, iterations=iterations, max_skew=max_skew, sweep_ns=sweep_ns)
