import json
import logging
import math
import os
import re
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


def _clock_period_ps(sdc_path):
    pattern = re.compile(r"create_clock\b.*?-period\s+([0-9eE+\-.]+)")
    with open(sdc_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _timer_units_to_ps(value, time_unit_seconds):
    value = _finite_or_none(value)
    time_unit_seconds = _finite_or_none(time_unit_seconds)
    if value is None or time_unit_seconds is None:
        return None
    return value * time_unit_seconds * 1e12


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


def run(config_path, iterations=1000, n=100, max_skew=50.0, output_path=None):
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
    params.useful_skew_weighting_flag = 0

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "opentimer"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer(params, placedb, learning_rate_value)

    timing_op = placer.op_collections.timing_op
    pos = placer.pos[0].data.clone().cpu()
    timing_op(pos)
    if timer.timer_engine == "opentimer":
        timing_op.timer.update_timing()

    if timer.timer_engine == "opentimer":
        time_unit = timing_op.timer.time_unit()
        final_tns = _timer_units_to_ps(timing_op.timer.report_tns_elw(split=1), time_unit)
        final_wns = _timer_units_to_ps(timing_op.timer.report_wns(split=1), time_unit)
    else:
        final_tns = _finite_or_none(timing_op.report_tns())
        final_wns = _finite_or_none(timing_op.report_wns())
    clock_period_ps = _clock_period_ps(params.sdc_input)

    graph = timing_op.timer.export_reg2reg_timing_graph(n=n, include_paths=False)
    baseline_setup = _count_violations(graph["edges"], "setup_slack")
    baseline_hold = _count_violations(graph["edges"], "hold_slack")
    skew = timing_op.timer.solve_useful_skew(n=n, max_skew=max_skew)

    summary = {
        "design": params.design_name(),
        "iterations": iterations,
        "timer_engine": timer.timer_engine,
        "clock_period_ps": clock_period_ps,
        "n": n,
        "max_skew_ps": max_skew,
        "final_placement_tns_ps": final_tns,
        "final_placement_wns_ps": final_wns,
        "final_max_violation_ps": None if final_wns is None else max(0.0, -float(final_wns)),
        "final_violation_over_period": None if clock_period_ps in (None, 0) or final_wns is None else max(0.0, -float(final_wns)) / clock_period_ps,
        "sampled_graph": {
            "path_counts": graph.get("path_counts"),
            "num_registers": graph["num_registers"],
            "num_edges": graph["num_edges"],
            "setup": baseline_setup,
            "hold": baseline_hold,
        },
        "single_useful_skew": {
            "skew_success": bool(skew.get("success")),
            "skew_status": int(skew.get("status", -1)),
            "skew_message": skew.get("message"),
            "skew_margin_ps": skew.get("margin"),
            "improvement_vs_sampled_setup_worst_ps": None
            if skew.get("margin") is None or baseline_setup["worst_slack"] is None
            else float(skew.get("margin")) - float(baseline_setup["worst_slack"]),
            "num_constraints": skew.get("num_constraints"),
        },
    }

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
            "usage: python run_postplace_single_skew_summary.py <json> [iterations] [n] [max_skew_ps] [output_json]"
        )

    config_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    n = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    max_skew = float(sys.argv[4]) if len(sys.argv) >= 5 else 50.0
    output_path = sys.argv[5] if len(sys.argv) >= 6 else None
    run(config_path, iterations=iterations, n=n, max_skew=max_skew, output_path=output_path)
