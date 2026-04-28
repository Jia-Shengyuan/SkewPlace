import json
import logging
import os
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

from dreamplace import Params
from dreamplace import PlaceDB
from dreamplace import Timer
from dreamplace import NonLinearPlace


def run(config_path):
    params = Params.Params()
    params.load(config_path)

    np.random.seed(params.random_seed)

    placedb = PlaceDB.PlaceDB()
    placedb(params)

    timer = Timer.Timer(timer_engine=getattr(params, "timer_engine", "heterosta"))
    timer(params, placedb)
    if timer.timer_engine == "opentimer":
        timer.update_timing()

    learning_rate_value = params.__dict__.get("global_place_stages")[0].get("learning_rate")
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    placer(params, placedb, learning_rate_value)

    timing_op = placer.op_collections.timing_op
    pos = placer.pos[0].data.clone().cpu()
    timing_op(pos)
    timing_op.timer.update_timing()

    graph = timing_op.timer.export_reg2reg_timing_graph(include_paths=True)
    skew = timing_op.timer.solve_useful_skew(max_skew=20.0)

    top_edges = []
    for edge in graph["edges"][:10]:
        top_edges.append({
            "launch_register": edge["launch_register"],
            "capture_register": edge["capture_register"],
            "setup_delay": edge.get("setup_delay"),
            "setup_constraint": edge.get("setup_constraint"),
            "setup_slack": edge.get("setup_slack"),
            "hold_delay": edge.get("hold_delay"),
            "hold_constraint": edge.get("hold_constraint"),
            "hold_slack": edge.get("hold_slack"),
        })

    summary = {
        "design": params.design_name(),
        "timer_engine": timer.timer_engine,
        "num_registers": graph["num_registers"],
        "num_edges": graph["num_edges"],
        "registers": graph["registers"],
        "top_edges": top_edges,
        "all_edges": graph["edges"],
        "skew_success": skew["success"],
        "skew_status": skew["status"],
        "skew_message": skew["message"],
        "skew_margin": skew["margin"],
        "skews": skew["skews"],
    }

    out_dir = os.path.join(ROOT, "results", "minitimer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "useful_skew_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "summary_path": out_path,
        "num_registers": summary["num_registers"],
        "num_edges": summary["num_edges"],
        "skew_success": summary["skew_success"],
        "skew_margin": summary["skew_margin"],
        "top_edges": summary["top_edges"],
        "skews": summary["skews"],
    }, indent=2))


if __name__ == "__main__":
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    if len(sys.argv) != 2:
        raise SystemExit("usage: python run_minitimer_useful_skew.py <json>")
    run(sys.argv[1])
