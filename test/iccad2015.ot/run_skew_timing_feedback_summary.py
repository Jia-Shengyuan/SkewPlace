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
            if value is not None:
                return _scalar_to_float(value)
        return None

    return {
        "objective": _last_value("objective"),
        "hpwl": _last_value("hpwl"),
        "overflow": _last_value("overflow"),
        "max_density": _last_value("max_density"),
        "tns": _last_value("tns"),
        "wns": _last_value("wns"),
    }


def _build_summary(params, timer, iterations, use_skew, skew_n, max_skew, clock_period_ps):
    return {
        "design": params.design_name(),
        "iterations": iterations,
        "clock_period_ps": clock_period_ps,
        "clock_frequency_mhz": None if clock_period_ps in (None, 0) else 1.0e6 / clock_period_ps,
        "useful_skew_weighting_flag": bool(use_skew),
        "useful_skew_weighting_n": skew_n,
        "useful_skew_max_skew": max_skew,
        "timer_engine": timer.timer_engine,
        "heterosta_use_cuda": bool(getattr(params, "heterosta_use_cuda", 0)),
    }


def _emit_summary(summary, output_path=None):
    text = json.dumps(summary, indent=2)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)


def run(config_path, iterations=520, use_skew=False, skew_n=100, max_skew=50.0, output_path=None):
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
    useful_skew_schedule_override = os.environ.get("DREAMPLACE_USEFUL_SKEW_SCHEDULE")
    if useful_skew_schedule_override is not None:
        params.useful_skew_schedule = useful_skew_schedule_override
    heterosta_use_cuda_override = os.environ.get("HETEROSTA_USE_CUDA")
    if heterosta_use_cuda_override is not None:
        params.heterosta_use_cuda = int(heterosta_use_cuda_override)

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    clock_period_ps = _clock_period_ps(params.sdc_input)
    summary = _build_summary(params, timer, iterations, use_skew, skew_n, max_skew, clock_period_ps)

    begin = time.time()
    try:
        metrics = placer(params, placedb, learning_rate_value)
        summary.update(
            {
                "runtime_sec": time.time() - begin,
                "stopped_due_to_error": False,
                "placement_metrics": _final_placement_metrics(metrics),
                "weighting_stats": getattr(placer.op_collections.timing_op, "last_useful_skew_weighting_stats", {}),
            }
        )
    except Exception as exc:
        summary.update(
            {
                "runtime_sec": time.time() - begin,
                "stopped_due_to_error": True,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "placement_metrics": None,
                "weighting_stats": getattr(placer.op_collections.timing_op, "last_useful_skew_weighting_stats", {}),
            }
        )

    _emit_summary(summary, output_path=output_path)
    return summary


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
            "usage: python run_skew_timing_feedback_summary.py <json> [iterations] [use_skew] [skew_n] [max_skew] [output_json]"
        )

    config_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 520
    use_skew = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else False
    skew_n = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    max_skew = float(sys.argv[5]) if len(sys.argv) >= 6 else 50.0
    output_path = sys.argv[6] if len(sys.argv) >= 7 else None
    summary = run(
        config_path,
        iterations=iterations,
        use_skew=use_skew,
        skew_n=skew_n,
        max_skew=max_skew,
        output_path=output_path,
    )
    if summary.get("stopped_due_to_error"):
        raise SystemExit(1)
