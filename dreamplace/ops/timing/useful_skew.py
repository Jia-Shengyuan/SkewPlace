import math

import numpy as np
from scipy.optimize import linprog


def _to_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _is_number(value):
    return value is not None and not math.isnan(value)


def _normalize_path(path):
    result = {}
    for key, value in path.items():
        if key == "points":
            points = []
            for point in value:
                normalized_point = {}
                for point_key, point_value in point.items():
                    normalized_point[_to_text(point_key)] = _to_text(point_value)
                points.append(normalized_point)
            result[_to_text(key)] = points
        else:
            result[_to_text(key)] = _to_text(value)
    return result


def _filter_test_paths(paths):
    normalized_paths = [_normalize_path(path) for path in paths]
    return [path for path in normalized_paths if path.get("endpoint_type") == "test"]


def _pin_suffix(point):
    return point.get("pin_name", "").rsplit(":", 1)[-1]


def _is_clock_pin(point):
    return _pin_suffix(point) in ("CK", "CLK", "CP", "G")


def _find_launch_point(points, capture_gate_name=""):
    search_points = points[:-1] if len(points) > 1 else points

    for index in range(len(search_points) - 1):
        point = search_points[index]
        next_point = search_points[index + 1]
        if (point.get("gate_name") and
                point.get("gate_name") == next_point.get("gate_name") and
                _is_clock_pin(point)):
            return next_point

    datapath_sources = [
        point for point in search_points
        if point.get("is_datapath_source") and point.get("gate_name") and not _is_clock_pin(point)
    ]
    if datapath_sources:
        return datapath_sources[0]

    gate_points = [
        point for point in search_points
        if point.get("gate_name") and point.get("gate_name") != capture_gate_name
    ]
    if gate_points:
        return gate_points[0]

    return None


def _register_name(point):
    if point is None:
        return ""
    gate_name = point.get("gate_name")
    if gate_name:
        return gate_name
    return point.get("pin_name", "")


def _register_info(point):
    return {
        "register_name": _register_name(point),
        "gate_name": point.get("gate_name", "") if point else "",
        "cell_name": point.get("cell_name", "") if point else "",
        "pin_name": point.get("pin_name", "") if point else "",
    }


def _path_to_edge(path, include_path):
    if path.get("endpoint_type") != "test":
        return None

    points = path.get("points", [])
    launch_point = _find_launch_point(points, path.get("capture_gate_name", ""))
    capture_point = {
        "gate_name": path.get("capture_gate_name", ""),
        "cell_name": path.get("capture_cell_name", ""),
        "pin_name": path.get("capture_pin_name", ""),
    }
    launch_info = _register_info(launch_point)
    capture_info = _register_info(capture_point)

    if (not launch_info["register_name"] or not capture_info["register_name"] or
            not launch_info["gate_name"] or not capture_info["gate_name"]):
        return None

    analysis_type = path.get("analysis_type")
    test_constraint = path.get("test_constraint")
    required_time = path.get("required_time")
    arrival_time = points[-1].get("arrival_time") if points else float("nan")
    path_delay = path.get("path_delay")
    slack = path.get("slack")

    edge = {
        "launch_register": launch_info["register_name"],
        "capture_register": capture_info["register_name"],
        "launch_gate_name": launch_info["gate_name"],
        "capture_gate_name": capture_info["gate_name"],
        "launch_cell_name": launch_info["cell_name"],
        "capture_cell_name": capture_info["cell_name"],
        "launch_pin_name": launch_info["pin_name"],
        "capture_pin_name": capture_info["pin_name"],
        "related_pin_name": path.get("related_pin_name", ""),
        "related_gate_name": path.get("related_gate_name", ""),
        "related_cell_name": path.get("related_cell_name", ""),
        "analysis_type": analysis_type,
        "endpoint_transition": path.get("endpoint_transition"),
        "slack": slack,
        "path_delay": path_delay,
        "arrival_time": arrival_time,
        "required_time": required_time,
        "test_constraint": test_constraint,
        "setup_delay": float("nan"),
        "hold_delay": float("nan"),
        "setup_constraint": float("nan"),
        "hold_constraint": float("nan"),
    }

    if analysis_type == "max":
        edge["setup_delay"] = path_delay
        edge["setup_constraint"] = test_constraint
    elif analysis_type == "min":
        edge["hold_delay"] = path_delay
        edge["hold_constraint"] = test_constraint
    else:
        return None

    if include_path:
        edge["path"] = path
    return edge


def _merge_edge(existing_edge, candidate_edge, include_paths):
    for field in (
            "launch_register",
            "capture_register",
            "launch_gate_name",
            "capture_gate_name",
            "launch_cell_name",
            "capture_cell_name",
            "launch_pin_name",
            "capture_pin_name",
            "related_pin_name",
            "related_gate_name",
            "related_cell_name"):
        if not existing_edge.get(field) and candidate_edge.get(field):
            existing_edge[field] = candidate_edge[field]

    if _is_number(candidate_edge.get("setup_delay")):
        if (not _is_number(existing_edge.get("setup_delay")) or
                candidate_edge["setup_delay"] > existing_edge["setup_delay"]):
            existing_edge["setup_delay"] = candidate_edge["setup_delay"]
            existing_edge["setup_constraint"] = candidate_edge.get("setup_constraint")
            existing_edge["setup_slack"] = candidate_edge.get("slack")
            existing_edge["setup_path_delay"] = candidate_edge.get("path_delay")
            existing_edge["setup_required_time"] = candidate_edge.get("required_time")
            existing_edge["setup_arrival_time"] = candidate_edge.get("arrival_time")
            existing_edge["setup_transition"] = candidate_edge.get("endpoint_transition")
            if include_paths:
                existing_edge["setup_path"] = candidate_edge.get("path")

    if _is_number(candidate_edge.get("hold_delay")):
        if (not _is_number(existing_edge.get("hold_delay")) or
                candidate_edge["hold_delay"] < existing_edge["hold_delay"]):
            existing_edge["hold_delay"] = candidate_edge["hold_delay"]
            existing_edge["hold_constraint"] = candidate_edge.get("hold_constraint")
            existing_edge["hold_slack"] = candidate_edge.get("slack")
            existing_edge["hold_path_delay"] = candidate_edge.get("path_delay")
            existing_edge["hold_required_time"] = candidate_edge.get("required_time")
            existing_edge["hold_arrival_time"] = candidate_edge.get("arrival_time")
            existing_edge["hold_transition"] = candidate_edge.get("endpoint_transition")
            if include_paths:
                existing_edge["hold_path"] = candidate_edge.get("path")


def build_reg2reg_timing_graph(paths, include_paths=False):
    normalized_paths = [_normalize_path(path) for path in paths]
    edge_map = {}
    register_map = {}

    for path in normalized_paths:
        edge = _path_to_edge(path, include_paths)
        if edge is None:
            continue

        key = (edge["launch_register"], edge["capture_register"])
        if key not in edge_map:
            edge_map[key] = {
                "launch_register": edge["launch_register"],
                "capture_register": edge["capture_register"],
                "launch_gate_name": edge["launch_gate_name"],
                "capture_gate_name": edge["capture_gate_name"],
                "launch_cell_name": edge["launch_cell_name"],
                "capture_cell_name": edge["capture_cell_name"],
                "launch_pin_name": edge["launch_pin_name"],
                "capture_pin_name": edge["capture_pin_name"],
                "related_pin_name": edge["related_pin_name"],
                "related_gate_name": edge["related_gate_name"],
                "related_cell_name": edge["related_cell_name"],
                "setup_delay": float("nan"),
                "hold_delay": float("nan"),
                "setup_constraint": float("nan"),
                "hold_constraint": float("nan"),
                "setup_slack": float("nan"),
                "hold_slack": float("nan"),
                "setup_path_delay": float("nan"),
                "hold_path_delay": float("nan"),
                "setup_required_time": float("nan"),
                "hold_required_time": float("nan"),
                "setup_arrival_time": float("nan"),
                "hold_arrival_time": float("nan"),
                "setup_transition": "",
                "hold_transition": "",
            }
            if include_paths:
                edge_map[key]["setup_path"] = None
                edge_map[key]["hold_path"] = None

        _merge_edge(edge_map[key], edge, include_paths)

        register_map.setdefault(edge["launch_register"], {
            "register_name": edge["launch_register"],
            "gate_name": edge["launch_gate_name"],
            "cell_name": edge["launch_cell_name"],
            "pin_name": edge["launch_pin_name"],
        })
        register_map.setdefault(edge["capture_register"], {
            "register_name": edge["capture_register"],
            "gate_name": edge["capture_gate_name"],
            "cell_name": edge["capture_cell_name"],
            "pin_name": edge["capture_pin_name"],
        })

    edges = list(edge_map.values())
    edges.sort(key=lambda edge: (
        math.inf if not _is_number(edge.get("setup_slack")) else edge["setup_slack"],
        edge["launch_register"],
        edge["capture_register"]))

    registers = list(register_map.values())
    registers.sort(key=lambda register: register["register_name"])

    return {
        "num_paths": len(normalized_paths),
        "num_registers": len(registers),
        "num_edges": len(edges),
        "registers": registers,
        "edges": edges,
    }


def build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=False):
    edge_map = {}
    register_map = {}
    num_paths = 0

    for split, paths in path_sets.items():
        normalized_paths = _filter_test_paths(paths)
        num_paths += len(normalized_paths)

        for path in normalized_paths:
            if split in ("max", "min"):
                path["analysis_type"] = split

            edge = _path_to_edge(path, include_paths)
            if edge is None:
                continue

            key = (edge["launch_register"], edge["capture_register"])
            if key not in edge_map:
                edge_map[key] = {
                    "launch_register": edge["launch_register"],
                    "capture_register": edge["capture_register"],
                    "launch_gate_name": edge["launch_gate_name"],
                    "capture_gate_name": edge["capture_gate_name"],
                    "launch_cell_name": edge["launch_cell_name"],
                    "capture_cell_name": edge["capture_cell_name"],
                    "launch_pin_name": edge["launch_pin_name"],
                    "capture_pin_name": edge["capture_pin_name"],
                    "related_pin_name": edge["related_pin_name"],
                    "related_gate_name": edge["related_gate_name"],
                    "related_cell_name": edge["related_cell_name"],
                    "setup_delay": float("nan"),
                    "hold_delay": float("nan"),
                    "setup_constraint": float("nan"),
                    "hold_constraint": float("nan"),
                    "setup_slack": float("nan"),
                    "hold_slack": float("nan"),
                    "setup_path_delay": float("nan"),
                    "hold_path_delay": float("nan"),
                    "setup_required_time": float("nan"),
                    "hold_required_time": float("nan"),
                    "setup_arrival_time": float("nan"),
                    "hold_arrival_time": float("nan"),
                    "setup_transition": "",
                    "hold_transition": "",
                }
                if include_paths:
                    edge_map[key]["setup_path"] = None
                    edge_map[key]["hold_path"] = None

            _merge_edge(edge_map[key], edge, include_paths)

            register_map.setdefault(edge["launch_register"], {
                "register_name": edge["launch_register"],
                "gate_name": edge["launch_gate_name"],
                "cell_name": edge["launch_cell_name"],
                "pin_name": edge["launch_pin_name"],
            })
            register_map.setdefault(edge["capture_register"], {
                "register_name": edge["capture_register"],
                "gate_name": edge["capture_gate_name"],
                "cell_name": edge["capture_cell_name"],
                "pin_name": edge["capture_pin_name"],
            })

    edges = list(edge_map.values())
    edges.sort(key=lambda edge: (
        math.inf if not _is_number(edge.get("setup_slack")) else edge["setup_slack"],
        edge["launch_register"],
        edge["capture_register"]))

    registers = list(register_map.values())
    registers.sort(key=lambda register: register["register_name"])

    return {
        "num_paths": num_paths,
        "num_registers": len(registers),
        "num_edges": len(edges),
        "registers": registers,
        "edges": edges,
        "path_counts": {split: len(paths) for split, paths in path_sets.items()},
    }


def export_reg2reg_timing_graph(timer, n=None, include_paths=False):
    path_sets = {
        "max": timer.report_test_paths_by_split("max", n=n),
        "min": timer.report_test_paths_by_split("min", n=n),
    }
    return build_reg2reg_timing_graph_from_split_paths(
        path_sets, include_paths=include_paths)


def solve_useful_skew(graph, max_skew=None):
    registers = graph.get("registers", [])
    edges = graph.get("edges", [])
    if not registers:
        return {
            "success": True,
            "status": 0,
            "message": "No registers found in timing graph",
            "margin": 0.0,
            "skews": {},
            "num_constraints": 0,
            "num_registers": 0,
        }

    register_names = [register["register_name"] for register in registers]
    register_index = {name: idx for idx, name in enumerate(register_names)}
    margin_index = len(register_names)
    variable_count = len(register_names) + 1

    a_ub = []
    b_ub = []
    kept_edges = []

    for edge in edges:
        launch_index = register_index[edge["launch_register"]]
        capture_index = register_index[edge["capture_register"]]

        if _is_number(edge.get("setup_slack")):
            row = np.zeros(variable_count, dtype=np.float64)
            row[launch_index] = 1.0
            row[capture_index] = -1.0
            row[margin_index] = 1.0
            # [Jsy] The previous prototype used delay/constraint directly, which
            # flipped the setup inequality and forced the optimized margin
            # negative even on slack-positive examples. Tighten against the
            # available slack so the LP margin matches timing intuition.
            b_ub.append(edge["setup_slack"])
            a_ub.append(row)
            kept_edges.append(edge)

        if _is_number(edge.get("hold_slack")):
            row = np.zeros(variable_count, dtype=np.float64)
            row[capture_index] = 1.0
            row[launch_index] = -1.0
            row[margin_index] = 1.0
            b_ub.append(edge["hold_slack"])
            a_ub.append(row)
            kept_edges.append(edge)

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[margin_index] = -1.0

    a_eq = np.zeros((1, variable_count), dtype=np.float64)
    a_eq[0, 0] = 1.0
    b_eq = np.zeros(1, dtype=np.float64)

    bounds = []
    for _ in register_names:
        if max_skew is None:
            bounds.append((None, None))
        else:
            bounds.append((-float(max_skew), float(max_skew)))
    bounds.append((None, None))

    result = linprog(
        c=objective,
        A_ub=np.array(a_ub, dtype=np.float64) if a_ub else None,
        b_ub=np.array(b_ub, dtype=np.float64) if b_ub else None,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs")

    skews = {}
    margin = float("nan")
    if result.x is not None:
        for index, name in enumerate(register_names):
            skews[name] = float(result.x[index])
        margin = float(result.x[margin_index])

    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "margin": margin,
        "skews": skews,
        "num_constraints": len(a_ub),
        "num_registers": len(register_names),
        "num_edges": len(edges),
        "max_skew": max_skew,
        "solver": "scipy.optimize.linprog(method=highs)",
    }


def solve_useful_skew_from_timer(timer, n=None, include_paths=False, max_skew=None):
    graph = export_reg2reg_timing_graph(timer, n=n, include_paths=include_paths)
    solution = solve_useful_skew(graph, max_skew=max_skew)
    solution["graph"] = graph
    return solution
