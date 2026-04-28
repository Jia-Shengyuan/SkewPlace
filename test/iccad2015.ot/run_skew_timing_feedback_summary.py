import json
import logging
import math
import os
import re
import sys
import time

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


def _scalar_to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _clock_period_ps(sdc_path):
    pattern = re.compile(r"create_clock\b.*?-period\s+([0-9eE+\-.]+)")
    with open(sdc_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


def _final_placement_metrics(metrics):
    if not metrics:
        return {}
    last = metrics[-1]
    return {
        "objective": _scalar_to_float(getattr(last, "objective", None)) if getattr(last, "objective", None) is not None else None,
        "hpwl": _scalar_to_float(getattr(last, "hpwl", None)) if getattr(last, "hpwl", None) is not None else None,
        "overflow": _scalar_to_float(getattr(last, "overflow", None)) if getattr(last, "overflow", None) is not None else None,
        "max_density": _scalar_to_float(getattr(last, "max_density", None)) if getattr(last, "max_density", None) is not None else None,
        "tns": _scalar_to_float(getattr(last, "tns", None)) if getattr(last, "tns", None) is not None else None,
        "wns": _scalar_to_float(getattr(last, "wns", None)) if getattr(last, "wns", None) is not None else None,
    }


def run(config_path, iterations=520, use_skew=False, skew_n=100, max_skew=50.0):
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
    params.useful_skew_weighting_flag = 1 if use_skew else 0
    params.useful_skew_weighting_n = skew_n
    params.useful_skew_max_skew = max_skew

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)

    begin = time.time()
    metrics = placer(params, placedb, learning_rate_value)
    elapsed = time.time() - begin
    clock_period_ps = _clock_period_ps(params.sdc_input)

    summary = {
        "design": params.design_name(),
        "iterations": iterations,
        "clock_period_ps": clock_period_ps,
        "clock_frequency_mhz": None if clock_period_ps in (None, 0) else 1.0e6 / clock_period_ps,
        "useful_skew_weighting_flag": bool(use_skew),
        "useful_skew_weighting_n": skew_n,
        "useful_skew_max_skew": max_skew,
        "timer_engine": timer.timer_engine,
        "runtime_sec": elapsed,
        "placement_metrics": _final_placement_metrics(metrics),
        "weighting_stats": getattr(placer.op_collections.timing_op, "last_useful_skew_weighting_stats", {}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stderr,
    )
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python run_skew_timing_feedback_summary.py <json> [iterations] [use_skew] [skew_n] [max_skew]"
        )

    config_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 520
    use_skew = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else False
    skew_n = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    max_skew = float(sys.argv[5]) if len(sys.argv) >= 6 else 50.0
    run(config_path, iterations=iterations, use_skew=use_skew, skew_n=skew_n, max_skew=max_skew)
