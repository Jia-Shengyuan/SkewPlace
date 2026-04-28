import json
import logging
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


def run(config_path, use_skew=False, iterations=520):
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
    params.useful_skew_weighting_n = 10
    params.useful_skew_max_skew = 20.0

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

    last_metric = metrics[-1] if metrics else None
    summary = {
        "use_skew": use_skew,
        "iterations": iterations,
        "final_hpwl": None if last_metric is None or getattr(last_metric, "hpwl", None) is None else float(last_metric.hpwl),
        "final_overflow": None if last_metric is None or getattr(last_metric, "overflow", None) is None else float(last_metric.overflow),
        "final_tns": None if last_metric is None or getattr(last_metric, "tns", None) is None else float(last_metric.tns),
        "final_wns": None if last_metric is None or getattr(last_metric, "wns", None) is None else float(last_metric.wns),
        "weighting_stats": getattr(placer.op_collections.timing_op, "last_useful_skew_weighting_stats", {}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run_minitimer_timing_feedback.py <json> [use_skew] [iterations]")
    config_path = sys.argv[1]
    use_skew = bool(int(sys.argv[2])) if len(sys.argv) >= 3 else False
    iterations = int(sys.argv[3]) if len(sys.argv) >= 4 else 520
    run(config_path, use_skew=use_skew, iterations=iterations)
