import json
import logging
import math
import os
import re
import sys
import time
import traceback

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
from dreamplace.ops.timing.useful_skew import _path_to_edge
from dreamplace.ops.timing.useful_skew import build_reg2reg_timing_graph_from_split_paths
from dreamplace.ops.timing.useful_skew import solve_useful_skew


def _abs_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def _clock_period_ps(sdc_path):
    pattern = re.compile(r"create_clock\b.*?-period\s+([0-9eE+\-.]+)")
    with open(sdc_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


def _finite(value):
    return value is not None and isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


def _count_slacks(values):
    worst = None
    tns = 0.0
    violations = 0
    total = 0
    for value in values:
        if not _finite(value):
            continue
        value = float(value)
        total += 1
        if worst is None or value < worst:
            worst = value
        if value < 0.0:
            violations += 1
            tns += value
    return {
        "worst_slack_ps": worst,
        "tns_ps": tns,
        "violating_paths": violations,
        "total_paths": total,
        "max_violation_ps": None if worst is None else max(0.0, -worst),
    }


def _final_placement_metrics(metrics):
    if not metrics:
        return {}

    flat_metrics = []

    def _collect(items):
        if items is None:
            return
        if isinstance(items, (list, tuple)):
            for item in items:
                _collect(item)
            return
        flat_metrics.append(items)

    _collect(metrics)
    if not flat_metrics:
        return {}

    def _last_value(field):
        for metric in reversed(flat_metrics):
            value = getattr(metric, field, None)
            if value is None:
                continue
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        return None

    return {
        "objective": _last_value("objective"),
        "weighted_hpwl": _last_value("hpwl"),
        "overflow": _last_value("overflow"),
        "max_density": _last_value("max_density"),
        "tns": _last_value("tns"),
        "wns": _last_value("wns"),
    }


def _true_hpwl(placedb, pos):
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    elif hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()

    x = pos[: placedb.num_nodes]
    y = pos[placedb.num_nodes : placedb.num_nodes * 2]
    wl = 0.0
    for net_id, pins in enumerate(placedb.net2pin_map):
        if len(pins) <= 1:
            continue
        nodes = placedb.pin2node_map[pins]
        pin_x = x[nodes] + placedb.pin_offset_x[pins]
        pin_y = y[nodes] + placedb.pin_offset_y[pins]
        wl += float(np.max(pin_x) - np.min(pin_x) + np.max(pin_y) - np.min(pin_y))
    return wl


def _full_setup_sta_from_timer(timing_op):
    setup_wns = timing_op.report_wns()
    setup_tns = timing_op.report_tns()
    setup_wns = float(setup_wns) if _finite(setup_wns) else None
    setup_tns = float(setup_tns) if _finite(setup_tns) else None
    return {
        "worst_slack_ps": setup_wns,
        "tns_ps": setup_tns,
        "violating_paths": None,
        "total_paths": None,
        "max_violation_ps": None if setup_wns is None else max(0.0, -setup_wns),
        "source": "heterosta_full_design_report",
    }


def _path_register_key(path):
    edge = _path_to_edge(path, include_path=False)
    if edge is None:
        return "", ""
    return edge.get("launch_register", ""), edge.get("capture_register", "")


def _adjusted_path_slacks(paths, skews, is_setup):
    adjusted = []
    matched_raw = []
    matched_adjusted = []
    unmatched_raw = []
    unmatched_examples = []
    unmatched = 0
    for path in paths:
        raw_slack = path.get("slack")
        if not _finite(raw_slack):
            continue
        raw_slack = float(raw_slack)
        launch, capture = _path_register_key(path)
        if not launch or not capture:
            unmatched += 1
            unmatched_raw.append(raw_slack)
            adjusted.append(raw_slack)
            if len(unmatched_examples) < 10:
                points = path.get("points", [])
                first = points[0] if points else {}
                last = points[-1] if points else {}
                unmatched_examples.append({
                    "slack_ps": raw_slack,
                    "start_pin_name": path.get("start_pin_name", ""),
                    "end_pin_name": path.get("end_pin_name", ""),
                    "launch_pin_name": path.get("launch_pin_name", ""),
                    "capture_pin_name": path.get("capture_pin_name", ""),
                    "launch_gate_name": path.get("launch_gate_name", ""),
                    "capture_gate_name": path.get("capture_gate_name", ""),
                    "first_point_pin_name": first.get("pin_name", ""),
                    "first_point_gate_name": first.get("gate_name", ""),
                    "last_point_pin_name": last.get("pin_name", ""),
                    "last_point_gate_name": last.get("gate_name", ""),
                    "num_points": path.get("num_points"),
                    "analysis_type": path.get("analysis_type"),
                })
            continue
        launch_skew = float(skews.get(launch, 0.0))
        capture_skew = float(skews.get(capture, 0.0))
        if is_setup:
            adjusted_slack = raw_slack + capture_skew - launch_skew
        else:
            adjusted_slack = raw_slack + launch_skew - capture_skew
        matched_raw.append(raw_slack)
        matched_adjusted.append(adjusted_slack)
        adjusted.append(adjusted_slack)
    return {
        "adjusted": adjusted,
        "matched_raw": matched_raw,
        "matched_adjusted": matched_adjusted,
        "unmatched_raw": unmatched_raw,
        "unmatched": unmatched,
        "unmatched_examples": unmatched_examples,
    }


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _stable_close(a, b, atol=1e-3):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= atol


def _extract_stats_for_paths(path_sets, max_skew):
    graph = build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=False)
    skew = solve_useful_skew(graph, max_skew=max_skew)
    skews = skew.get("skews", {})

    raw_setup_slacks = [path.get("slack") for path in path_sets["max"] if _finite(path.get("slack"))]
    raw_hold_slacks = [path.get("slack") for path in path_sets["min"] if _finite(path.get("slack"))]
    adjusted_setup = _adjusted_path_slacks(path_sets["max"], skews, is_setup=True)
    adjusted_hold = _adjusted_path_slacks(path_sets["min"], skews, is_setup=False)

    return {
        "graph": graph,
        "skew": skew,
        "raw_setup": _count_slacks(raw_setup_slacks),
        "raw_hold": _count_slacks(raw_hold_slacks),
        "raw_setup_matched": _count_slacks(adjusted_setup["matched_raw"]),
        "raw_hold_matched": _count_slacks(adjusted_hold["matched_raw"]),
        "raw_setup_unmatched": _count_slacks(adjusted_setup["unmatched_raw"]),
        "raw_hold_unmatched": _count_slacks(adjusted_hold["unmatched_raw"]),
        "adjusted_setup": _count_slacks(adjusted_setup["adjusted"]),
        "adjusted_hold": _count_slacks(adjusted_hold["adjusted"]),
        "adjusted_setup_matched": _count_slacks(adjusted_setup["matched_adjusted"]),
        "adjusted_hold_matched": _count_slacks(adjusted_hold["matched_adjusted"]),
        "adjusted_setup_unmatched": _count_slacks(adjusted_setup["unmatched_raw"]),
        "adjusted_hold_unmatched": _count_slacks(adjusted_hold["unmatched_raw"]),
        "unmatched_setup": adjusted_setup["unmatched"],
        "unmatched_hold": adjusted_hold["unmatched"],
        "unmatched_setup_examples": adjusted_setup["unmatched_examples"],
        "unmatched_hold_examples": adjusted_hold["unmatched_examples"],
    }


def _full_graph_stats(timing_op, max_skew):
    graph = timing_op.timer.export_full_reg2reg_timing_graph(include_paths=False)
    skew = solve_useful_skew(graph, max_skew=max_skew)
    return {
        "graph": graph,
        "skew": skew,
    }


def _full_path_stats(timing_op, max_skew):
    path_sets = {
        "max": timing_op.timer.report_all_timing_paths_by_split("max"),
        "min": timing_op.timer.report_all_timing_paths_by_split("min"),
    }
    return _extract_stats_for_paths(path_sets, max_skew=max_skew)


def _select_path_count(timing_op, max_skew, initial_count=1000, max_count=None):
    evaluations = []
    previous = None
    path_count = max(1, int(initial_count))
    if max_count is None:
        max_count = int(os.environ.get("DREAMPLACE_SUMMARY_MAX_PATHS", "20000"))
    while True:
        path_sets = {
            "max": timing_op.timer.report_timing_paths_by_split("max", n=path_count),
            "min": timing_op.timer.report_timing_paths_by_split("min", n=path_count),
        }
        current = _extract_stats_for_paths(path_sets, max_skew=max_skew)
        evaluations.append({
            "path_count": path_count,
            "max_paths": len(path_sets["max"]),
            "min_paths": len(path_sets["min"]),
            "raw_setup_worst_ps": current["raw_setup"]["worst_slack_ps"],
            "raw_setup_tns_ps": current["raw_setup"]["tns_ps"],
            "raw_setup_matched_worst_ps": current["raw_setup_matched"]["worst_slack_ps"],
            "raw_setup_matched_tns_ps": current["raw_setup_matched"]["tns_ps"],
            "raw_setup_unmatched_worst_ps": current["raw_setup_unmatched"]["worst_slack_ps"],
            "raw_setup_unmatched_tns_ps": current["raw_setup_unmatched"]["tns_ps"],
            "skew_setup_worst_ps": current["adjusted_setup"]["worst_slack_ps"],
            "skew_setup_tns_ps": current["adjusted_setup"]["tns_ps"],
            "skew_setup_matched_worst_ps": current["adjusted_setup_matched"]["worst_slack_ps"],
            "skew_setup_matched_tns_ps": current["adjusted_setup_matched"]["tns_ps"],
        })

        if previous is not None:
            converged = (
                _stable_close(previous["raw_setup"]["worst_slack_ps"], current["raw_setup"]["worst_slack_ps"]) and
                _stable_close(previous["raw_setup"]["tns_ps"], current["raw_setup"]["tns_ps"]) and
                _stable_close(previous["raw_setup_matched"]["worst_slack_ps"], current["raw_setup_matched"]["worst_slack_ps"]) and
                _stable_close(previous["raw_setup_matched"]["tns_ps"], current["raw_setup_matched"]["tns_ps"]) and
                _stable_close(previous["adjusted_setup"]["worst_slack_ps"], current["adjusted_setup"]["worst_slack_ps"]) and
                _stable_close(previous["adjusted_setup"]["tns_ps"], current["adjusted_setup"]["tns_ps"]) and
                _stable_close(previous["adjusted_setup_matched"]["worst_slack_ps"], current["adjusted_setup_matched"]["worst_slack_ps"]) and
                _stable_close(previous["adjusted_setup_matched"]["tns_ps"], current["adjusted_setup_matched"]["tns_ps"])
            )
            if converged:
                return path_count, path_sets, current, evaluations, True

        if path_count >= max_count:
            return path_count, path_sets, current, evaluations, False

        previous = current
        path_count = min(max_count, path_count * 2)


def run(
    config_path,
    iterations=1000,
    max_skew=0.0,
    eps_paths=1000,
    output_path=None,
    placement_mode="single-skew",
    placement_max_skew=None,
    skew_n=100,
):
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
    params.timer_engine = "heterosta"
    params.heterosta_use_cuda = int(os.environ.get("HETEROSTA_USE_CUDA", "1"))
    use_full_export = os.environ.get("DREAMPLACE_SUMMARY_FULL_EXPORT", "0") == "1"
    params.useful_skew_weighting_flag = 0
    params.useful_skew_weighting_n = int(skew_n)
    params.useful_skew_schedule = "fixed"
    placement_target_skew = float(max_skew if placement_max_skew is None else placement_max_skew)
    params.useful_skew_max_skew = placement_target_skew
    if placement_mode in ("dynamic-skew", "growing-skew"):
        params.useful_skew_weighting_flag = 1
    if placement_mode == "growing-skew":
        params.useful_skew_schedule = "growing"

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine="heterosta")
    timer(params, placedb)

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)

    clock_period_ps = _clock_period_ps(params.sdc_input)
    summary = {
        "design": params.design_name(),
        "iterations": iterations,
        "timer_engine": "heterosta",
        "heterosta_use_cuda": bool(getattr(params, "heterosta_use_cuda", 0)),
        "clock_period_ps": clock_period_ps,
        "analysis_max_skew_ps": float(max_skew),
        "placement_mode": placement_mode,
        "placement_max_skew_ps": placement_target_skew,
        "placement_skew_n": int(skew_n),
        "path_count_request": int(eps_paths),
    }

    begin = time.time()
    try:
        metrics = placer(params, placedb, learning_rate_value)
        runtime_sec = time.time() - begin

        timing_op = placer.op_collections.timing_op
        pos = placer.pos[0].data.clone()
        if getattr(params, "gpu", 0):
            pos = pos.cuda()
        timing_op(pos)

        selected_path_count, path_sets, stats, convergence, path_count_converged = _select_path_count(
            timing_op,
            max_skew=max_skew,
            initial_count=eps_paths,
        )
        sampled_graph = stats["graph"]
        raw_setup = _full_setup_sta_from_timer(timing_op)
        raw_hold = stats["raw_hold"]
        adjusted_setup = stats["adjusted_setup"]
        adjusted_hold = stats["adjusted_hold"]
        unmatched_setup = stats["unmatched_setup"]
        unmatched_hold = stats["unmatched_hold"]
        placement_metrics = _final_placement_metrics(metrics)
        placement_metrics["hpwl"] = _true_hpwl(placedb, placer.pos[0].data)

        full_graph = None
        full_path_stats = None
        skew_solution = stats["skew"]
        if use_full_export:
            full_graph_stats = _full_graph_stats(timing_op, max_skew=max_skew)
            full_graph = full_graph_stats["graph"]
            full_path_stats = _full_path_stats(timing_op, max_skew=max_skew)
            skew_solution = full_path_stats["skew"]
            adjusted_setup = full_path_stats["adjusted_setup"]
            adjusted_hold = full_path_stats["adjusted_hold"]

        summary.update({
            "runtime_sec": runtime_sec,
            "stopped_due_to_error": False,
            "placement_metrics": placement_metrics,
            "path_count_selected": selected_path_count,
            "path_count_converged": path_count_converged,
            "path_count_convergence": convergence,
            "summary_full_export_enabled": use_full_export,
            "path_coverage": {
                "max_paths": len(path_sets["max"]),
                "min_paths": len(path_sets["min"]),
                "sampled_graph_registers": sampled_graph.get("num_registers"),
                "sampled_graph_edges": sampled_graph.get("num_edges"),
                "full_graph_registers": None if full_graph is None else full_graph.get("num_registers"),
                "full_graph_edges": None if full_graph is None else full_graph.get("num_edges"),
                "unmatched_setup_paths": unmatched_setup,
                "unmatched_hold_paths": unmatched_hold,
            },
            "raw_sta": {
                "setup": {
                    **raw_setup,
                    "violation_over_period": _ratio(raw_setup.get("max_violation_ps"), clock_period_ps),
                },
                "hold": {
                    **raw_hold,
                    "violation_over_period": _ratio(raw_hold.get("max_violation_ps"), clock_period_ps),
                },
            },
            "sampled_path_sta": {
                "setup": {
                    **stats["raw_setup"],
                    "violation_over_period": _ratio(stats["raw_setup"].get("max_violation_ps"), clock_period_ps),
                },
                "hold": {
                    **stats["raw_hold"],
                    "violation_over_period": _ratio(stats["raw_hold"].get("max_violation_ps"), clock_period_ps),
                },
            },
            "matched_path_sta": {
                "raw_setup": {
                    **stats["raw_setup_matched"],
                    "violation_over_period": _ratio(stats["raw_setup_matched"].get("max_violation_ps"), clock_period_ps),
                },
                "skewed_setup": {
                    **stats["adjusted_setup_matched"],
                    "violation_over_period": _ratio(stats["adjusted_setup_matched"].get("max_violation_ps"), clock_period_ps),
                },
                "raw_hold": {
                    **stats["raw_hold_matched"],
                    "violation_over_period": _ratio(stats["raw_hold_matched"].get("max_violation_ps"), clock_period_ps),
                },
                "skewed_hold": {
                    **stats["adjusted_hold_matched"],
                    "violation_over_period": _ratio(stats["adjusted_hold_matched"].get("max_violation_ps"), clock_period_ps),
                },
            },
            "unmatched_path_sta": {
                "setup": {
                    **stats["raw_setup_unmatched"],
                    "violation_over_period": _ratio(stats["raw_setup_unmatched"].get("max_violation_ps"), clock_period_ps),
                    "examples": stats["unmatched_setup_examples"],
                },
                "hold": {
                    **stats["raw_hold_unmatched"],
                    "violation_over_period": _ratio(stats["raw_hold_unmatched"].get("max_violation_ps"), clock_period_ps),
                    "examples": stats["unmatched_hold_examples"],
                },
            },
            "skew_solution": {
                "success": bool(skew_solution.get("success")),
                "status": int(skew_solution.get("status", -1)),
                "message": skew_solution.get("message"),
                "margin_ps": skew_solution.get("margin"),
                "num_constraints": skew_solution.get("num_constraints"),
                "num_registers": skew_solution.get("num_registers"),
                "num_edges": skew_solution.get("num_edges"),
                "solver": skew_solution.get("solver"),
                "source": "full_path_export" if use_full_export else "converged_top_paths",
            },
            "skew_aware_sta": {
                "setup": {
                    **adjusted_setup,
                    "violation_over_period": _ratio(adjusted_setup.get("max_violation_ps"), clock_period_ps),
                },
                "hold": {
                    **adjusted_hold,
                    "violation_over_period": _ratio(adjusted_hold.get("max_violation_ps"), clock_period_ps),
                },
            },
            "full_path_coverage": {
                "max_paths": None if full_path_stats is None else full_path_stats["graph"].get("path_counts", {}).get("max"),
                "min_paths": None if full_path_stats is None else full_path_stats["graph"].get("path_counts", {}).get("min"),
                "unmatched_setup_paths": None if full_path_stats is None else full_path_stats["unmatched_setup"],
                "unmatched_hold_paths": None if full_path_stats is None else full_path_stats["unmatched_hold"],
            },
        })
    except Exception as exc:
        summary.update({
            "runtime_sec": time.time() - begin,
            "stopped_due_to_error": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })

    text = json.dumps(summary, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)
    return summary


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python run_heterosta_skew_aware_sta_summary.py <json> [analysis_max_skew_ps] [iterations] [path_count] [placement_mode] [placement_max_skew_ps] [skew_n] [output_json]"
        )

    config_path = sys.argv[1]
    max_skew = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.0
    iterations = int(sys.argv[3]) if len(sys.argv) >= 4 else 1000
    path_count = int(sys.argv[4]) if len(sys.argv) >= 5 else 1000
    placement_mode = sys.argv[5] if len(sys.argv) >= 6 else "single-skew"
    placement_max_skew = float(sys.argv[6]) if len(sys.argv) >= 7 else None
    skew_n = int(sys.argv[7]) if len(sys.argv) >= 8 else 100
    output_path = sys.argv[8] if len(sys.argv) >= 9 else None
    run(
        config_path,
        iterations=iterations,
        max_skew=max_skew,
        eps_paths=path_count,
        output_path=output_path,
        placement_mode=placement_mode,
        placement_max_skew=placement_max_skew,
        skew_n=skew_n,
    )
