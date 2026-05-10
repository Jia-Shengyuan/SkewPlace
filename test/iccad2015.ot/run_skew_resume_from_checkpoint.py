import json
import logging
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


def _clock_period_ps(sdc_path):
    pattern = re.compile(r"create_clock\b.*?-period\s+([0-9eE+\-.]+)")
    with open(sdc_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


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


def run(
    config_path,
    checkpoint_path,
    start_iteration,
    total_iterations=1000,
    placement_mode="dynamic-skew",
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
    params.global_place_stages[0]["iteration"] = int(total_iterations) - int(start_iteration)
    params.global_place_iteration_offset = int(start_iteration)
    params.global_place_total_iterations = int(total_iterations)
    params.gift_init_flag = 0
    params.useful_skew_weighting_flag = 1 if placement_mode in ("dynamic-skew", "growing-skew") else 0
    params.useful_skew_weighting_n = int(skew_n)
    params.useful_skew_schedule = "growing" if placement_mode == "growing-skew" else "fixed"
    params.useful_skew_max_skew = float(placement_max_skew)
    params.timer_engine = "heterosta"
    params.heterosta_use_cuda = int(os.environ.get("HETEROSTA_USE_CUDA", "1"))

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine="heterosta")
    timer(params, placedb)

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer.load(params, placedb, checkpoint_path)

    begin = time.time()
    metrics = placer(params, placedb, learning_rate_value)
    runtime_sec = time.time() - begin

    summary = {
        "design": params.design_name(),
        "checkpoint_path": checkpoint_path,
        "start_iteration": int(start_iteration),
        "total_iterations": int(total_iterations),
        "remaining_iterations": int(total_iterations) - int(start_iteration),
        "placement_mode": placement_mode,
        "placement_max_skew_ps": float(placement_max_skew),
        "placement_skew_n": int(skew_n),
        "clock_period_ps": _clock_period_ps(params.sdc_input),
        "runtime_sec": runtime_sec,
        "placement_metrics": _final_placement_metrics(metrics),
    }
    summary["placement_metrics"]["hpwl"] = _true_hpwl(placedb, placer.pos[0].data)

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
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python run_skew_resume_from_checkpoint.py <json> <checkpoint.pklz> <start_iteration> [total_iterations] [placement_mode] [placement_max_skew_ps] [skew_n] [output_json]"
        )

    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    start_iteration = int(sys.argv[3])
    total_iterations = int(sys.argv[4]) if len(sys.argv) >= 5 else 1000
    placement_mode = sys.argv[5] if len(sys.argv) >= 6 else "dynamic-skew"
    placement_max_skew = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.0
    skew_n = int(sys.argv[7]) if len(sys.argv) >= 8 else 100
    output_path = sys.argv[8] if len(sys.argv) >= 9 else None
    run(
        config_path,
        checkpoint_path,
        start_iteration=start_iteration,
        total_iterations=total_iterations,
        placement_mode=placement_mode,
        placement_max_skew=placement_max_skew,
        skew_n=skew_n,
        output_path=output_path,
    )
