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


def _net_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _top_changed_nets(placedb, before_weights, before_criticality, limit=20):
    rows = []
    for net_id, name in enumerate(placedb.net_names):
        weight_before = float(before_weights[net_id])
        weight_after = float(placedb.net_weights[net_id])
        crit_before = float(before_criticality[net_id])
        crit_after = float(placedb.net_criticality[net_id])
        delta = weight_after - weight_before
        if abs(delta) <= 1e-12 and abs(crit_after - crit_before) <= 1e-12:
            continue
        rows.append({
            "net_id": int(net_id),
            "net_name": _net_name(name),
            "weight_before": weight_before,
            "weight_after": weight_after,
            "weight_delta": delta,
            "criticality_before": crit_before,
            "criticality_after": crit_after,
        })
    rows.sort(key=lambda row: abs(row["weight_delta"]), reverse=True)
    return rows[:limit], len(rows)


def run(config_path, checkpoint_path, use_skew=False, n=100, max_skew=50.0):
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
    params.useful_skew_weighting_flag = 1 if use_skew else 0
    params.useful_skew_weighting_n = n
    params.useful_skew_max_skew = max_skew

    np.random.seed(params.random_seed)

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

    before_weights = placedb.net_weights.copy()
    before_criticality = placedb.net_criticality.copy()
    timing_op.update_net_weights(max_net_weight=placedb.max_net_weight, n=n)
    top_changed, changed_count = _top_changed_nets(placedb, before_weights, before_criticality)

    summary = {
        "use_skew": use_skew,
        "checkpoint_path": checkpoint_path,
        "n": n,
        "max_skew": max_skew,
        "changed_net_count": changed_count,
        "top_changed_nets": top_changed,
        "weighting_stats": getattr(timing_op, "last_useful_skew_weighting_stats", {}),
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
            "usage: python debug_skew_weighting_checkpoint.py <json> <checkpoint.pklz> [use_skew] [n] [max_skew]"
        )
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    use_skew = bool(int(sys.argv[3])) if len(sys.argv) >= 4 else False
    n = int(sys.argv[4]) if len(sys.argv) >= 5 else 100
    max_skew = float(sys.argv[5]) if len(sys.argv) >= 6 else 50.0
    run(config_path, checkpoint_path, use_skew=use_skew, n=n, max_skew=max_skew)
