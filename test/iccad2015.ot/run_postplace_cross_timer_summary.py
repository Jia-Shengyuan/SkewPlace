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


def _true_hpwl(placedb, pos):
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    elif hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()

    x = pos[: placedb.num_nodes]
    y = pos[placedb.num_nodes : placedb.num_nodes * 2]
    wl = 0.0
    for pins in placedb.net2pin_map:
        if len(pins) <= 1:
            continue
        nodes = placedb.pin2node_map[pins]
        pin_x = x[nodes] + placedb.pin_offset_x[pins]
        pin_y = y[nodes] + placedb.pin_offset_y[pins]
        wl += float(np.max(pin_x) - np.min(pin_x) + np.max(pin_y) - np.min(pin_y))
    return wl


def _metric_value(metric, field):
    value = getattr(metric, field, None)
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _final_placement_metrics(metrics):
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

    metric = flat_metrics[-1]
    return {
        "objective": _metric_value(metric, "objective"),
        "weighted_hpwl": _metric_value(metric, "hpwl"),
        "overflow": _metric_value(metric, "overflow"),
        "max_density": _metric_value(metric, "max_density"),
        "tns": _metric_value(metric, "tns"),
        "wns": _metric_value(metric, "wns"),
    }


def _timer_summary_with_existing_timing_op(timer_engine, timing_op, pos):
    timing_pos = pos.clone().cpu() if timer_engine == "opentimer" else pos.clone()
    if timer_engine == "heterosta" and hasattr(timing_op, "use_cuda") and bool(getattr(timing_op, "use_cuda", False)):
        timing_pos = timing_pos.cuda()
    timing_op(timing_pos)
    if timer_engine == "opentimer":
        timing_op.timer.update_timing()

    if timer_engine == "opentimer":
        time_unit = timing_op.timer.time_unit()
        setup_tns_ps = timing_op.timer.report_tns_elw(split=1) / (time_unit * 1e17)
        setup_wns_ps = timing_op.timer.report_wns(split=1) / (time_unit * 1e15)
    else:
        setup_tns_ps = timing_op.report_tns()
        setup_wns_ps = timing_op.report_wns()

    setup_wns_ps = float(setup_wns_ps) if _finite(setup_wns_ps) else None
    setup_tns_ps = float(setup_tns_ps) if _finite(setup_tns_ps) else None
    return {
        "timer_engine": timer_engine,
        "setup_wns_ps": setup_wns_ps,
        "setup_tns_ps": setup_tns_ps,
        "max_violation_ps": None if setup_wns_ps is None else max(0.0, -setup_wns_ps),
    }


def _timer_summary(timer_engine, params, placedb, pos):
    timer = Timer.Timer(timer_engine=timer_engine)
    timer(params, placedb)
    if timer_engine == "opentimer":
        timer.update_timing()

    timing_op = NonLinearPlace.NonLinearPlace(params, placedb, timer).op_collections.timing_op
    return _timer_summary_with_existing_timing_op(timer_engine, timing_op, pos)


def run(
    config_path,
    iterations=1000,
    placement_mode="no-skew",
    placement_max_skew=0.0,
    skew_n=100,
    output_path=None,
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
    params.useful_skew_weighting_flag = 1 if placement_mode in ("dynamic-skew", "growing-skew") else 0
    params.useful_skew_weighting_n = int(skew_n)
    params.useful_skew_schedule = "growing" if placement_mode == "growing-skew" else "fixed"
    params.useful_skew_max_skew = float(placement_max_skew)

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    placement_timer = Timer.Timer(timer_engine="heterosta")
    placement_timer(params, placedb)
    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, placement_timer)

    clock_period_ps = _clock_period_ps(params.sdc_input)
    summary = {
        "design": params.design_name(),
        "iterations": iterations,
        "placement_mode": placement_mode,
        "placement_max_skew_ps": float(placement_max_skew),
        "placement_skew_n": int(skew_n),
        "clock_period_ps": clock_period_ps,
    }

    begin = time.time()
    try:
        metrics = placer(params, placedb, learning_rate_value)
        pos = placer.pos[0].data.clone()
        heterosta_timing_op = placer.op_collections.timing_op
        placement_metrics = _final_placement_metrics(metrics)
        placement_metrics["hpwl"] = _true_hpwl(placedb, pos)

        summary.update({
            "runtime_sec": time.time() - begin,
            "stopped_due_to_error": False,
            "placement_metrics": placement_metrics,
            "postplace_retiming": {
                "heterosta": _timer_summary_with_existing_timing_op("heterosta", heterosta_timing_op, pos),
                "opentimer": _timer_summary("opentimer", params, placedb, pos),
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
            "usage: python run_postplace_cross_timer_summary.py <json> [iterations] [placement_mode] [placement_max_skew_ps] [skew_n] [output_json]"
        )

    config_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    placement_mode = sys.argv[3] if len(sys.argv) >= 4 else "no-skew"
    placement_max_skew = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.0
    skew_n = int(sys.argv[5]) if len(sys.argv) >= 6 else 100
    output_path = sys.argv[6] if len(sys.argv) >= 7 else None
    run(
        config_path,
        iterations=iterations,
        placement_mode=placement_mode,
        placement_max_skew=placement_max_skew,
        skew_n=skew_n,
        output_path=output_path,
    )
