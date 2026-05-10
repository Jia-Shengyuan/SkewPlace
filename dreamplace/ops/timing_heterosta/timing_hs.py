import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import dreamplace.ops.timing_heterosta.timing_heterosta_cpp as timing_hs_cpp
import logging
import pdb

from dreamplace.ops.timing.useful_skew import build_reg2reg_timing_graph_from_split_paths
from dreamplace.ops.timing.useful_skew import export_reg2reg_timing_graph
from dreamplace.ops.timing.useful_skew import solve_useful_skew
from dreamplace.ops.timing.useful_skew import solve_useful_skew_from_timer


def _is_number(value):
    return value is not None and isinstance(value, (int, float, np.floating)) and np.isfinite(float(value))


def _pin2net_array(pin2net_map):
    if hasattr(pin2net_map, 'detach'):
        return pin2net_map.detach().cpu().numpy()
    if hasattr(pin2net_map, 'cpu'):
        return pin2net_map.cpu().numpy()
    return np.asarray(pin2net_map)


def _unique_path_net_ids(path, pin2net_map):
    pin2net = _pin2net_array(pin2net_map)
    net_ids = []
    seen = set()
    for point in path.get("points", []):
        pin_id = point.get("pin_id")
        if pin_id is None:
            continue
        pin_id = int(pin_id)
        if pin_id < 0 or pin_id >= len(pin2net):
            continue
        net_id = int(pin2net[pin_id])
        if net_id < 0 or net_id in seen:
            continue
        seen.add(net_id)
        net_ids.append(net_id)
    return net_ids


def _is_timing_number(value):
    return value is not None and isinstance(value, (int, float, np.floating)) and np.isfinite(float(value))

def _convert_pin_direction_to_numeric(pin_direct_strings):
    """
    @brief Convert pin direction strings to numeric encoding
    @param pin_direct_strings numpy array of byte strings (e.g., b'INPUT', b'OUTPUT')
    @return numpy array of uint8 values (0=INPUT, 1=OUTPUT, 2=INOUT)
    """
    direction_map = {
        b'INPUT': 0,
        b'OUTPUT': 1, 
        b'INOUT': 2,
        b'OUTPUT_TRISTATE': 1,  # Treat as OUTPUT
        b'UNKNOWN': 2 # Heterosta treats any number other than 0 and 1 as unknown.
    }
    
    # Handle both string and byte string inputs
    result = np.zeros(len(pin_direct_strings), dtype=np.uint8)
    for i, direction in enumerate(pin_direct_strings):
        # Convert string to bytes if needed
        if isinstance(direction, str):
            direction = direction.encode('utf-8')
        result[i] = direction_map.get(direction, 0)  # Default to INPUT if unknown
    
    return result

def _package_dreamplace_mappings(placedb):
    """
    @brief Package DREAMPlace mappings into a dictionary for C++ interface
    @param placedb the placement database containing the mappings
    @return dictionary containing only the necessary mappings as torch tensors
    """
    return {
        # Only include mappings that are actually used by the C++ code
        'pin2net_map': torch.from_numpy(placedb.pin2net_map),
        'pin2node_map': torch.from_numpy(placedb.pin2node_map),
        'pin_direct': torch.from_numpy(_convert_pin_direction_to_numeric(placedb.pin_direct)),
        'num_terminal_NIs': torch.tensor(placedb.num_terminal_NIs, dtype=torch.int32),
    }


class TimingIO(Function):
    """
    @brief The timer IO class for HeteroSTA integration
    HeteroSTA reads some external files like liberty libraries, SDC, etc. 
    The file reading and parsing will be done only once exactly after 
    the initialization of placement database.
    """
    @staticmethod
    def read(params, placedb):
        """
        @brief read design and store in placement database
        @param params the parameters defined in json
        @param placedb the placement database for netlist integration
        """
        # Build argument string for HeteroSTA
        args = "DREAMPLACE"  # First argument should be non-empty
        
        if "early_lib_input" in params.__dict__ and params.early_lib_input:
            args += " --early_lib_input %s" % (params.early_lib_input)
        if "late_lib_input" in params.__dict__ and params.late_lib_input:
            args += " --late_lib_input %s" % (params.late_lib_input)
        if "lib_input" in params.__dict__ and params.lib_input:
            # Use same library for both early and late if only one specified
            args += " --lib_input %s" % (params.lib_input)
        if "sdc_input" in params.__dict__ and params.sdc_input:
            args += " --sdc_input %s" % (params.sdc_input)
        # Note: verilog_input is not used since we reuse netlist from PlaceDB
        
        # Package DREAMPlace mappings to ensure data consistency
        dreamplace_mappings = _package_dreamplace_mappings(placedb)
        
        return timing_hs_cpp.io_forward(
            args.split(' '), 
            placedb.rawdb,
            dreamplace_mappings
        )


class TimingOptFunction(Function):
    @staticmethod
    def forward(ctx, timer, pos, 
                num_pins,
                wire_resistance_per_micron,
                wire_capacitance_per_micron,
                scale_factor, lef_unit, def_unit,
                pin_pos_op,
                slacks_rf=None,
                ignore_net_degree=np.iinfo(np.int32).max, use_cuda=False):
        """
        @brief compute timing analysis using HeteroSTA
        @param timer the HeteroSTA timer object
        @param pos node/cell locations (x array, y array), NOT pin locations
        @param num_pins total number of pins in the design
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param slacks_rf optional output tensor for pin slacks (rise/fall)
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        
        # Handle CUDA/CPU mode consistency
        pos_is_cuda = pos.is_cuda
        if use_cuda and not pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CUDA for GPU timing analysis")
            pos = pos.cuda()
        elif not use_cuda and pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CPU for CPU timing analysis")
            pos = pos.cpu()
        
        # HeteroSTA RC extraction expects pin coordinates indexed in the same
        # order used to build the NetlistDB.
        logging.info("HeteroSTA forward: before pin_pos_op (use_cuda=%s pos_is_cuda=%s pos_numel=%d)", bool(use_cuda), pos.is_cuda, pos.numel())
        pin_pos = pin_pos_op(pos)
        logging.info("HeteroSTA forward: after pin_pos_op (pin_pos_is_cuda=%s pin_pos_numel=%d)", pin_pos.is_cuda, pin_pos.numel())
        
        # Create slack output tensor if requested
        if slacks_rf is None:
            slacks_rf = torch.Tensor()
        
        logging.info("HeteroSTA forward: before timing_hs_cpp.forward")
        timing_hs_cpp.forward(
            timer,
            pin_pos,
            num_pins,
            wire_resistance_per_micron,
            wire_capacitance_per_micron,
            scale_factor, lef_unit, def_unit,
            slacks_rf,
            ignore_net_degree, bool(use_cuda))
        logging.info("HeteroSTA forward: after timing_hs_cpp.forward")
        
        return torch.zeros(num_pins)

class TimingOpt(nn.Module):
    def __init__(self, timer, net_names, pin_names, flat_netpin,
                 netpin_start, net_name2id_map, pin_name2id_map,
                 pin2node_map, pin_offset_x, pin_offset_y,
                 pin2net_map, net_criticality, net_criticality_deltas,
                 net_weights, net_weight_deltas,
                 wire_resistance_per_micron,
                 wire_capacitance_per_micron,
                 momentum_decay_factor,
                 scale_factor, lef_unit, def_unit,
                 pin_pos_op,
                 ignore_net_degree, use_cuda=False,
                 useful_skew_weighting_flag=0,
                 useful_skew_weighting_n=100,
                 useful_skew_max_skew=50.0):
        """
        @brief Initialize the feedback module for HeteroSTA timing analysis
        @param timer the HeteroSTA timer object
        @param net_names the name of each net
        @param pin_names the name of each pin
        @param flat_netpin the net2pin map logic (1d flatten array)
        @param netpin_start the start indices in the flat_netpin
        @param net_name2id_map the net name to id map
        @param pin_name2id_map the pin name to id map
        @param pin2node_map the 1d array pin2node map
        @param pin_offset_x pin offset x to its node
        @param pin_offset_y pin offset y to its node
        @param pin2net_map the pin to net mapping array
        @param net_criticality net criticality value
        @param net_criticality_deltas net criticality delta value
        @param net_weights net weights of placedb
        @param net_weight_deltas the increment of net weights
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param momentum_decay_factor the decay factor in momentum iteration
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        super(TimingOpt, self).__init__()
        self.timer = timer
        # Only keep parameters actually needed for timing analysis
        self.flat_netpin = flat_netpin  # numpy array - will be converted when needed
        self.netpin_start = netpin_start  # numpy array - will be converted when needed
        self.net_name2id_map = net_name2id_map
        self.pin_name2id_map = pin_name2id_map
        self.pin2net_map = pin2net_map  # Store the provided pin2net_map (can be numpy or tensor)

        # Store tensor references directly (already on correct device - CPU or GPU)
        self.net_criticality = net_criticality  # torch.Tensor
        self.net_criticality_deltas = net_criticality_deltas  # torch.Tensor
        self.net_weights = net_weights  # torch.Tensor (shared with data_collections)
        self.net_weight_deltas = net_weight_deltas  # torch.Tensor
        self.net_criticality_cpu = net_criticality.detach().cpu().clone()
        self.net_criticality_deltas_cpu = net_criticality_deltas.detach().cpu().clone()
        self.net_weights_cpu = net_weights.detach().cpu().clone()
        self.net_weight_deltas_cpu = net_weight_deltas.detach().cpu().clone()
        self.net_criticality_cpu = net_criticality.detach().cpu().clone()
        self.net_criticality_deltas_cpu = net_criticality_deltas.detach().cpu().clone()
        self.net_weights_cpu = net_weights.detach().cpu().clone()
        self.net_weight_deltas_cpu = net_weight_deltas.detach().cpu().clone()
        self.wire_resistance_per_micron = wire_resistance_per_micron
        self.wire_capacitance_per_micron = wire_capacitance_per_micron
        self.momentum_decay_factor = momentum_decay_factor
        self.useful_skew_weighting_flag = bool(useful_skew_weighting_flag)
        self.useful_skew_weighting_n = int(useful_skew_weighting_n)
        if useful_skew_max_skew is None:
            self.useful_skew_max_skew = None
        else:
            self.useful_skew_max_skew = float(useful_skew_max_skew)
            if self.useful_skew_max_skew < 0:
                self.useful_skew_max_skew = None
        self.last_useful_skew_weighting_stats = {}
        
        # The scale factor is important, together with the lef/def unit.
        # Since we require the actual wire-length evaluation (microns) to
        # enable timing analysis, the parameters should be passed into
        # the cpp core functions.
        self.scale_factor = scale_factor
        self.lef_unit = lef_unit
        self.def_unit = def_unit
        self.ignore_net_degree = ignore_net_degree
        self.use_cuda = use_cuda
        
        # Calculate pin and net counts directly from data
        self.num_pins = len(pin_names)
        self.num_nets = len(net_names)
        
        self.degree_map = self.netpin_start[1:] - self.netpin_start[:-1]
        
        
        # Store the pin_pos_op passed from outside
        self.pin_pos_op = pin_pos_op

    def _active_weight_tensors(self):
        if self.use_cuda:
            return (
                self.net_criticality,
                self.net_criticality_deltas,
                self.net_weights,
                self.net_weight_deltas,
            )
        return (
            self.net_criticality_cpu,
            self.net_criticality_deltas_cpu,
            self.net_weights_cpu,
            self.net_weight_deltas_cpu,
        )

    def _sync_active_weights_to_placement(self):
        if self.use_cuda:
            return
        self.net_criticality.copy_(self.net_criticality_cpu.to(self.net_criticality.device))
        self.net_criticality_deltas.copy_(self.net_criticality_deltas_cpu.to(self.net_criticality_deltas.device))
        self.net_weights.copy_(self.net_weights_cpu.to(self.net_weights.device))
        self.net_weight_deltas.copy_(self.net_weight_deltas_cpu.to(self.net_weight_deltas.device))

    def _active_weight_tensors(self):
        if self.use_cuda:
            return (
                self.net_criticality,
                self.net_criticality_deltas,
                self.net_weights,
                self.net_weight_deltas,
            )
        return (
            self.net_criticality_cpu,
            self.net_criticality_deltas_cpu,
            self.net_weights_cpu,
            self.net_weight_deltas_cpu,
        )

    def _sync_active_weights_to_placement(self):
        if self.use_cuda:
            return
        self.net_criticality.copy_(self.net_criticality_cpu.to(self.net_criticality.device))
        self.net_criticality_deltas.copy_(self.net_criticality_deltas_cpu.to(self.net_criticality_deltas.device))
        self.net_weights.copy_(self.net_weights_cpu.to(self.net_weights.device))
        self.net_weight_deltas.copy_(self.net_weight_deltas_cpu.to(self.net_weight_deltas.device))

    def forward(self, pos):
        """
        @brief call HeteroSTA timing forward function
        @param pos node/cell coordinates (x and y arrays), pin coordinates will be calculated internally
        """
        
        result = TimingOptFunction.apply(
            self.timer.raw_timer,  # Pass the raw C++ timer object
            pos,  # The coordinates
            self.num_pins,  # Pass the pin count directly
            self.wire_resistance_per_micron,
            self.wire_capacitance_per_micron,
            self.scale_factor, self.lef_unit, self.def_unit,
            self.pin_pos_op,  # Pass the pin_pos_op
            None,  # slacks_rf (optional output tensor)
            self.ignore_net_degree, self.use_cuda)
        
        return result
    

    def update_net_weights(self, max_net_weight=np.inf, n=1):
        """
        @brief update net weights of placedb using HeteroSTA simplified algorithm
        @param max_net_weight the maximum net weight in timing opt
        @param n the maximum number of paths to be reported
        """
        if self.useful_skew_weighting_flag:
            return self._update_net_weights_useful_skew(max_net_weight=max_net_weight, n=n)

        try:
            active_net_criticality, active_net_criticality_deltas, active_net_weights, active_net_weight_deltas = self._active_weight_tensors()
            # Convert flat_netpin and netpin_start to tensors
            flat_netpin_t = torch.from_numpy(self.flat_netpin)
            netpin_start_t = torch.from_numpy(self.netpin_start)

            # Handle pin2net_map (can be numpy or tensor)
            if hasattr(self.pin2net_map, 'cpu'):
                pin2net_map_t = self.pin2net_map
            else:
                pin2net_map_t = torch.from_numpy(np.array(self.pin2net_map, dtype=np.int32))

            # Convert degree_map to tensor
            degree_map_t = torch.from_numpy(self.degree_map)

            # Move to CUDA if needed
            if self.use_cuda:
                flat_netpin_t = flat_netpin_t.cuda()
                netpin_start_t = netpin_start_t.cuda()
                if not pin2net_map_t.is_cuda:
                    pin2net_map_t = pin2net_map_t.cuda()
                degree_map_t = degree_map_t.cuda()

            # Call C++ update_net_weights with num_nets and num_pins instead of maps
            # Note: self.net_criticality, self.net_weights are already torch.Tensors on correct device
            # They will be modified in-place by C++
            timing_hs_cpp.update_net_weights(
                self.timer.raw_timer, n,
                self.num_nets, self.num_pins,  # Pass counts instead of maps
                flat_netpin_t,
                netpin_start_t,
                pin2net_map_t,
                active_net_criticality,
                active_net_criticality_deltas,
                active_net_weights,
                active_net_weight_deltas,
                degree_map_t,
                max_net_weight, self.momentum_decay_factor,
                self.ignore_net_degree, bool(self.use_cuda))

            self._sync_active_weights_to_placement()

        except Exception as e:
            logging.error(f"HeteroSTA net weight update failed: {e}")
            raise  # Re-raise the exception so caller knows it failed

    def _update_net_weights_useful_skew(self, max_net_weight=np.inf, n=1):
        effective_n = max(1, min(int(n), self.useful_skew_weighting_n))
        path_sets = {
            "max": self.report_timing_paths_by_split("max", n=effective_n),
            "min": self.report_timing_paths_by_split("min", n=effective_n),
        }
        graph = build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=True)
        solution = solve_useful_skew(graph, max_skew=self.useful_skew_max_skew)

        stats = {
            "enabled": True,
            "sampled_n": effective_n,
            "path_counts": graph.get("path_counts"),
            "num_registers": graph.get("num_registers"),
            "num_edges": graph.get("num_edges"),
            "max_skew": self.useful_skew_max_skew,
            "skew_success": bool(solution.get("success")),
            "skew_status": int(solution.get("status", -1)),
            "skew_message": solution.get("message"),
            "skew_margin": solution.get("margin"),
        }

        if not solution.get("success") or not _is_number(solution.get("margin")):
            logging.warning(
                "HeteroSTA useful-skew weighting fallback to raw weighting: success=%s status=%s message=%s",
                solution.get("success"),
                solution.get("status"),
                solution.get("message"),
            )
            stats["fallback_to_raw"] = True
            self.last_useful_skew_weighting_stats = stats
            return self._update_net_weights_raw(max_net_weight=max_net_weight, n=n)

        skews = solution.get("skews", {})
        net_slack_deltas = {}
        raw_setup_wns = None
        adjusted_setup_wns = None

        for edge in graph.get("edges", []):
            setup_slack = edge.get("setup_slack")
            if not _is_timing_number(setup_slack):
                continue

            launch_skew = float(skews.get(edge["launch_register"], 0.0))
            capture_skew = float(skews.get(edge["capture_register"], 0.0))
            adjusted_setup_slack = float(setup_slack) + capture_skew - launch_skew

            raw_setup_wns = float(setup_slack) if raw_setup_wns is None else min(raw_setup_wns, float(setup_slack))
            adjusted_setup_wns = adjusted_setup_slack if adjusted_setup_wns is None else min(adjusted_setup_wns, adjusted_setup_slack)

            setup_path = edge.get("setup_path")
            if not setup_path:
                continue

            for net_id in _unique_path_net_ids(setup_path, self.pin2net_map):
                previous = adjusted_net_slacks.get(net_id)
                if previous is None or adjusted_setup_slack < previous:
                    adjusted_net_slacks[net_id] = adjusted_setup_slack

        finite_max_weight = _is_number(max_net_weight)
        updated_nets = 0
        device = self.net_weights.device if hasattr(self.net_weights, "device") else None
        dtype = self.net_weights.dtype if hasattr(self.net_weights, "dtype") else None
        degree_map_t = torch.from_numpy(self.degree_map)
        if device is not None:
            degree_map_t = degree_map_t.to(device)

        for net_id in range(self.num_nets):
            if self.degree_map[net_id] > self.ignore_net_degree:
                continue

            nc = 0.0
            adjusted_slack = adjusted_net_slacks.get(net_id)
            if (
                adjusted_slack is not None
                and adjusted_setup_wns is not None
                and adjusted_setup_wns < 0.0
                and adjusted_slack < 0.0
            ):
                nc = max(0.0, float(adjusted_slack) / float(adjusted_setup_wns))
                updated_nets += 1

            if hasattr(self.net_criticality[net_id], "item"):
                current = float(self.net_criticality[net_id].item())
            else:
                current = float(self.net_criticality[net_id])
            new_criticality = (
                np.power(1.0 + current, self.momentum_decay_factor)
                * np.power(1.0 + nc, 1.0 - self.momentum_decay_factor)
                - 1.0
            )
            value = torch.tensor(new_criticality, device=device, dtype=dtype) if device is not None else new_criticality
            self.net_criticality[net_id] = value
            self.net_weights[net_id] *= (1.0 + new_criticality)
            if finite_max_weight and float(self.net_weights[net_id].item()) > float(max_net_weight):
                self.net_weights[net_id] = torch.tensor(float(max_net_weight), device=device, dtype=dtype)

        stats["raw_setup_wns"] = raw_setup_wns
        stats["adjusted_setup_wns"] = adjusted_setup_wns
        stats["affected_nets"] = updated_nets
        stats["fallback_to_raw"] = False
        self.last_useful_skew_weighting_stats = stats
        self._sync_active_weights_to_placement()
        logging.info(
            "HeteroSTA useful-skew weighting: n=%d edges=%d margin=%.3f raw_setup_wns=%s adjusted_setup_wns=%s affected_nets=%d max_skew=%s",
            effective_n,
            graph.get("num_edges", 0),
            float(solution.get("margin")),
            "None" if raw_setup_wns is None else f"{raw_setup_wns:.3f}",
            "None" if adjusted_setup_wns is None else f"{adjusted_setup_wns:.3f}",
            updated_nets,
            self.useful_skew_max_skew,
        )
        return 0

    def _update_net_weights_raw(self, max_net_weight=np.inf, n=1):
        old_flag = self.useful_skew_weighting_flag
        self.useful_skew_weighting_flag = False
        try:
            return self.update_net_weights(max_net_weight=max_net_weight, n=n)
        finally:
            self.useful_skew_weighting_flag = old_flag
    
    def write_spef(self,file_path):
        return self.timer.raw_timer.write_spef(file_path)
    
    
    def report_wns_tns(self):
        """
        @brief report WNS and TNS in the design
        """
        return timing_hs_cpp.report_wns_tns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_wns(self):
        """
        @brief report WNS in the design
        """
        return timing_hs_cpp.report_wns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_tns(self):
        """
        @brief report TNS in the design
        """
        return timing_hs_cpp.report_tns(
            self.timer.raw_timer, bool(self.use_cuda))

    def update_timing(self):
        """
        This is a no-op since timing is updated in forward()
        """
        pass

    def time_unit(self):
        """
        @brief report time unit in the design
        """
        return 1e-12  # HeteroSTA uses picosecond as unit



    def dump_paths_setup_to_file(self, num_paths, nworst, file_path, use_cuda=False):
        """
        @brief dump timing paths to file using HeteroSTA
        @param num_paths number of paths to dump
        @param nworst number of worst paths per endpoint
        @param file_path output file path
        @param use_cuda whether to use CUDA
        """
        return self.timer.raw_timer.dump_paths_setup_to_file(num_paths, nworst, file_path, use_cuda)

    def report_timing_paths_by_split(self, split, n=1):
        """@brief report HeteroSTA timing paths with OpenTimer-like dictionaries."""
        return self.timer.raw_timer.report_timing_paths_by_split(split, n, self.use_cuda)

    def report_all_timing_paths_by_split(self, split):
        """@brief report all available HeteroSTA timing paths for one split."""
        return self.timer.raw_timer.report_all_timing_paths_by_split(split, self.use_cuda)

    def export_reg2reg_timing_graph(self, n=None, include_paths=False):
        """@brief export a register-to-register timing graph from HeteroSTA paths."""
        path_sets = {
            "max": self.report_timing_paths_by_split("max", n=n or 1),
            "min": self.report_timing_paths_by_split("min", n=n or 1),
        }
        return build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=include_paths)

    def export_full_reg2reg_timing_graph(self, include_paths=False):
        """@brief export a register-to-register timing graph from all available HeteroSTA paths."""
        path_sets = {
            "max": self.report_all_timing_paths_by_split("max"),
            "min": self.report_all_timing_paths_by_split("min"),
        }
        return build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=include_paths)
    
import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import dreamplace.ops.timing_heterosta.timing_heterosta_cpp as timing_hs_cpp
import logging
import pdb

def _convert_pin_direction_to_numeric(pin_direct_strings):
    """
    @brief Convert pin direction strings to numeric encoding
    @param pin_direct_strings numpy array of byte strings (e.g., b'INPUT', b'OUTPUT')
    @return numpy array of uint8 values (0=INPUT, 1=OUTPUT, 2=INOUT)
    """
    direction_map = {
        b'INPUT': 0,
        b'OUTPUT': 1, 
        b'INOUT': 2,
        b'OUTPUT_TRISTATE': 1,  # Treat as OUTPUT
        b'UNKNOWN': 2 # Heterosta treats any number other than 0 and 1 as unknown.
    }
    
    # Handle both string and byte string inputs
    result = np.zeros(len(pin_direct_strings), dtype=np.uint8)
    for i, direction in enumerate(pin_direct_strings):
        # Convert string to bytes if needed
        if isinstance(direction, str):
            direction = direction.encode('utf-8')
        result[i] = direction_map.get(direction, 0)  # Default to INPUT if unknown
    
    return result

def _package_dreamplace_mappings(placedb):
    """
    @brief Package DREAMPlace mappings into a dictionary for C++ interface
    @param placedb the placement database containing the mappings
    @return dictionary containing only the necessary mappings as torch tensors
    """
    return {
        # Only include mappings that are actually used by the C++ code
        'pin2net_map': torch.from_numpy(placedb.pin2net_map),
        'pin2node_map': torch.from_numpy(placedb.pin2node_map),
        'pin_direct': torch.from_numpy(_convert_pin_direction_to_numeric(placedb.pin_direct)),
        'num_terminal_NIs': torch.tensor(placedb.num_terminal_NIs, dtype=torch.int32),
    }


class TimingIO(Function):
    """
    @brief The timer IO class for HeteroSTA integration
    HeteroSTA reads some external files like liberty libraries, SDC, etc. 
    The file reading and parsing will be done only once exactly after 
    the initialization of placement database.
    """
    @staticmethod
    def read(params, placedb):
        """
        @brief read design and store in placement database
        @param params the parameters defined in json
        @param placedb the placement database for netlist integration
        """
        # Build argument string for HeteroSTA
        args = "DREAMPLACE"  # First argument should be non-empty
        
        if "early_lib_input" in params.__dict__ and params.early_lib_input:
            args += " --early_lib_input %s" % (params.early_lib_input)
        if "late_lib_input" in params.__dict__ and params.late_lib_input:
            args += " --late_lib_input %s" % (params.late_lib_input)
        if "lib_input" in params.__dict__ and params.lib_input:
            # Use same library for both early and late if only one specified
            args += " --lib_input %s" % (params.lib_input)
        if "sdc_input" in params.__dict__ and params.sdc_input:
            args += " --sdc_input %s" % (params.sdc_input)
        # Note: verilog_input is not used since we reuse netlist from PlaceDB
        
        # Package DREAMPlace mappings to ensure data consistency
        dreamplace_mappings = _package_dreamplace_mappings(placedb)
        
        return timing_hs_cpp.io_forward(
            args.split(' '), 
            placedb.rawdb,
            dreamplace_mappings
        )


class TimingOptFunction(Function):
    @staticmethod
    def forward(ctx, timer, pos, 
                num_pins,
                wire_resistance_per_micron,
                wire_capacitance_per_micron,
                scale_factor, lef_unit, def_unit,
                pin_pos_op,
                slacks_rf=None,
                ignore_net_degree=np.iinfo(np.int32).max, use_cuda=False):
        """
        @brief compute timing analysis using HeteroSTA
        @param timer the HeteroSTA timer object
        @param pos node/cell locations (x array, y array), NOT pin locations
        @param num_pins total number of pins in the design
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param slacks_rf optional output tensor for pin slacks (rise/fall)
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        
        # Handle CUDA/CPU mode consistency
        pos_is_cuda = pos.is_cuda
        if use_cuda and not pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CUDA for GPU timing analysis")
            pos = pos.cuda()
        elif not use_cuda and pos_is_cuda:
            logging.info("HeteroSTA: Converting pos to CPU for CPU timing analysis")
            pos = pos.cpu()
        
        # HeteroSTA RC extraction expects pin coordinates indexed in the same
        # order used to build the NetlistDB.
        logging.info("HeteroSTA forward: before pin_pos_op (use_cuda=%s pos_is_cuda=%s pos_numel=%d)", bool(use_cuda), pos.is_cuda, pos.numel())
        pin_pos = pin_pos_op(pos)
        logging.info("HeteroSTA forward: after pin_pos_op (pin_pos_is_cuda=%s pin_pos_numel=%d)", pin_pos.is_cuda, pin_pos.numel())
        
        # Create slack output tensor if requested
        if slacks_rf is None:
            slacks_rf = torch.Tensor()
        
        logging.info("HeteroSTA forward: before timing_hs_cpp.forward")
        timing_hs_cpp.forward(
            timer,
            pin_pos,
            num_pins,
            wire_resistance_per_micron,
            wire_capacitance_per_micron,
            scale_factor, lef_unit, def_unit,
            slacks_rf,
            ignore_net_degree, bool(use_cuda))
        logging.info("HeteroSTA forward: after timing_hs_cpp.forward")
        
        return torch.zeros(num_pins)

class TimingOpt(nn.Module):
    def __init__(self, timer, net_names, pin_names, flat_netpin,
                 netpin_start, net_name2id_map, pin_name2id_map,
                 pin2node_map, pin_offset_x, pin_offset_y,
                 pin2net_map, net_criticality, net_criticality_deltas,
                 net_weights, net_weight_deltas,
                 wire_resistance_per_micron,
                 wire_capacitance_per_micron,
                 momentum_decay_factor,
                 scale_factor, lef_unit, def_unit,
                 pin_pos_op,
                 ignore_net_degree, use_cuda=False,
                 useful_skew_weighting_flag=0,
                 useful_skew_weighting_n=100,
                 useful_skew_max_skew=50.0):
        """
        @brief Initialize the feedback module for HeteroSTA timing analysis
        @param timer the HeteroSTA timer object
        @param net_names the name of each net
        @param pin_names the name of each pin
        @param flat_netpin the net2pin map logic (1d flatten array)
        @param netpin_start the start indices in the flat_netpin
        @param net_name2id_map the net name to id map
        @param pin_name2id_map the pin name to id map
        @param pin2node_map the 1d array pin2node map
        @param pin_offset_x pin offset x to its node
        @param pin_offset_y pin offset y to its node
        @param pin2net_map the pin to net mapping array
        @param net_criticality net criticality value
        @param net_criticality_deltas net criticality delta value
        @param net_weights net weights of placedb
        @param net_weight_deltas the increment of net weights
        @param wire_resistance_per_micron unit-length resistance value
        @param wire_capacitance_per_micron unit-length capacitance value
        @param momentum_decay_factor the decay factor in momentum iteration
        @param scale_factor the scaling factor to be applied to the design
        @param lef_unit the unit distance microns defined in the LEF file
        @param def_unit the unit distance microns defined in the DEF file 
        @param pin_pos_op the pin position operator to compute pin locations from cell locations
        @param ignore_net_degree the degree threshold
        @param use_cuda whether to use CUDA for computation
        """
        super(TimingOpt, self).__init__()
        self.timer = timer
        # Only keep parameters actually needed for timing analysis
        self.flat_netpin = flat_netpin  # numpy array - will be converted when needed
        self.netpin_start = netpin_start  # numpy array - will be converted when needed
        self.net_name2id_map = net_name2id_map
        self.pin_name2id_map = pin_name2id_map
        self.pin2net_map = pin2net_map  # Store the provided pin2net_map (can be numpy or tensor)

        # Store tensor references directly (already on correct device - CPU or GPU)
        self.net_criticality = net_criticality  # torch.Tensor
        self.net_criticality_deltas = net_criticality_deltas  # torch.Tensor
        self.net_weights = net_weights  # torch.Tensor (shared with data_collections)
        self.net_weight_deltas = net_weight_deltas  # torch.Tensor
        self.net_criticality_cpu = net_criticality.detach().cpu().clone()
        self.net_criticality_deltas_cpu = net_criticality_deltas.detach().cpu().clone()
        self.net_weights_cpu = net_weights.detach().cpu().clone()
        self.net_weight_deltas_cpu = net_weight_deltas.detach().cpu().clone()
        self.wire_resistance_per_micron = wire_resistance_per_micron
        self.wire_capacitance_per_micron = wire_capacitance_per_micron
        self.momentum_decay_factor = momentum_decay_factor
        self.useful_skew_weighting_flag = bool(useful_skew_weighting_flag)
        self.useful_skew_weighting_n = int(useful_skew_weighting_n)
        if useful_skew_max_skew is None:
            self.useful_skew_max_skew = None
        else:
            self.useful_skew_max_skew = float(useful_skew_max_skew)
            if self.useful_skew_max_skew < 0:
                self.useful_skew_max_skew = None
        self.last_useful_skew_weighting_stats = {}
        
        # The scale factor is important, together with the lef/def unit.
        # Since we require the actual wire-length evaluation (microns) to
        # enable timing analysis, the parameters should be passed into
        # the cpp core functions.
        self.scale_factor = scale_factor
        self.lef_unit = lef_unit
        self.def_unit = def_unit
        self.ignore_net_degree = ignore_net_degree
        self.use_cuda = use_cuda
        
        # Calculate pin and net counts directly from data
        self.num_pins = len(pin_names)
        self.num_nets = len(net_names)
        
        self.degree_map = self.netpin_start[1:] - self.netpin_start[:-1]
        
        
        # Store the pin_pos_op passed from outside
        self.pin_pos_op = pin_pos_op

    def _active_weight_tensors(self):
        if self.use_cuda:
            return (
                self.net_criticality,
                self.net_criticality_deltas,
                self.net_weights,
                self.net_weight_deltas,
            )
        return (
            self.net_criticality_cpu,
            self.net_criticality_deltas_cpu,
            self.net_weights_cpu,
            self.net_weight_deltas_cpu,
        )

    def _sync_active_weights_to_placement(self):
        if self.use_cuda:
            return
        self.net_criticality.copy_(self.net_criticality_cpu.to(self.net_criticality.device))
        self.net_criticality_deltas.copy_(self.net_criticality_deltas_cpu.to(self.net_criticality_deltas.device))
        self.net_weights.copy_(self.net_weights_cpu.to(self.net_weights.device))
        self.net_weight_deltas.copy_(self.net_weight_deltas_cpu.to(self.net_weight_deltas.device))

    def forward(self, pos):
        """
        @brief call HeteroSTA timing forward function
        @param pos node/cell coordinates (x and y arrays), pin coordinates will be calculated internally
        """
        
        result = TimingOptFunction.apply(
            self.timer.raw_timer,  # Pass the raw C++ timer object
            pos,  # The coordinates
            self.num_pins,  # Pass the pin count directly
            self.wire_resistance_per_micron,
            self.wire_capacitance_per_micron,
            self.scale_factor, self.lef_unit, self.def_unit,
            self.pin_pos_op,  # Pass the pin_pos_op
            None,  # slacks_rf (optional output tensor)
            self.ignore_net_degree, self.use_cuda)
        
        return result
    

    def update_net_weights(self, max_net_weight=np.inf, n=1):
        """
        @brief update net weights of placedb using HeteroSTA simplified algorithm
        @param max_net_weight the maximum net weight in timing opt
        @param n the maximum number of paths to be reported
        """
        if self.useful_skew_weighting_flag:
            return self._update_net_weights_useful_skew(max_net_weight=max_net_weight, n=n)

        try:
            active_net_criticality, active_net_criticality_deltas, active_net_weights, active_net_weight_deltas = self._active_weight_tensors()
            # Convert flat_netpin and netpin_start to tensors
            flat_netpin_t = torch.from_numpy(self.flat_netpin)
            netpin_start_t = torch.from_numpy(self.netpin_start)

            # Handle pin2net_map (can be numpy or tensor)
            if hasattr(self.pin2net_map, 'cpu'):
                pin2net_map_t = self.pin2net_map
            else:
                pin2net_map_t = torch.from_numpy(np.array(self.pin2net_map, dtype=np.int32))

            # Convert degree_map to tensor
            degree_map_t = torch.from_numpy(self.degree_map)

            # Move to CUDA if needed
            if self.use_cuda:
                flat_netpin_t = flat_netpin_t.cuda()
                netpin_start_t = netpin_start_t.cuda()
                if not pin2net_map_t.is_cuda:
                    pin2net_map_t = pin2net_map_t.cuda()
                degree_map_t = degree_map_t.cuda()

            # Call C++ update_net_weights with num_nets and num_pins instead of maps
            # Note: self.net_criticality, self.net_weights are already torch.Tensors on correct device
            # They will be modified in-place by C++
            timing_hs_cpp.update_net_weights(
                self.timer.raw_timer, n,
                self.num_nets, self.num_pins,  # Pass counts instead of maps
                flat_netpin_t,
                netpin_start_t,
                pin2net_map_t,
                active_net_criticality,
                active_net_criticality_deltas,
                active_net_weights,
                active_net_weight_deltas,
                degree_map_t,
                max_net_weight, self.momentum_decay_factor,
                self.ignore_net_degree, bool(self.use_cuda))

            self._sync_active_weights_to_placement()

        except Exception as e:
            logging.error(f"HeteroSTA net weight update failed: {e}")
            raise  # Re-raise the exception so caller knows it failed

    def _update_net_weights_useful_skew(self, max_net_weight=np.inf, n=1):
        effective_n = max(1, min(int(n), self.useful_skew_weighting_n))
        path_sets = {
            "max": self.report_timing_paths_by_split("max", n=effective_n),
            "min": self.report_timing_paths_by_split("min", n=effective_n),
        }
        graph = build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=True)
        solution = solve_useful_skew(graph, max_skew=self.useful_skew_max_skew)

        stats = {
            "enabled": True,
            "sampled_n": effective_n,
            "path_counts": graph.get("path_counts"),
            "num_registers": graph.get("num_registers"),
            "num_edges": graph.get("num_edges"),
            "max_skew": self.useful_skew_max_skew,
            "skew_success": bool(solution.get("success")),
            "skew_status": int(solution.get("status", -1)),
            "skew_message": solution.get("message"),
            "skew_margin": solution.get("margin"),
        }

        if not solution.get("success") or not _is_timing_number(solution.get("margin")):
            logging.warning(
                "HeteroSTA useful-skew weighting fallback to raw weighting: success=%s status=%s message=%s",
                solution.get("success"),
                solution.get("status"),
                solution.get("message"),
            )
            stats["fallback_to_raw"] = True
            self.last_useful_skew_weighting_stats = stats
            return self._update_net_weights_raw(max_net_weight=max_net_weight, n=n)

        skews = solution.get("skews", {})
        net_slack_deltas = {}
        raw_setup_wns = None
        adjusted_setup_wns = None

        for edge in graph.get("edges", []):
            setup_slack = edge.get("setup_slack")
            if not _is_timing_number(setup_slack):
                continue

            launch_skew = float(skews.get(edge["launch_register"], 0.0))
            capture_skew = float(skews.get(edge["capture_register"], 0.0))
            adjusted_setup_slack = float(setup_slack) + capture_skew - launch_skew

            raw_setup_wns = float(setup_slack) if raw_setup_wns is None else min(raw_setup_wns, float(setup_slack))
            adjusted_setup_wns = adjusted_setup_slack if adjusted_setup_wns is None else min(adjusted_setup_wns, adjusted_setup_slack)

            setup_path = edge.get("setup_path")
            if not setup_path:
                continue

            slack_delta = adjusted_setup_slack - float(setup_slack)

            for net_id in _unique_path_net_ids(setup_path, self.pin2net_map):
                previous = net_slack_deltas.get(net_id)
                if previous is None or slack_delta < previous:
                    net_slack_deltas[net_id] = slack_delta

        adjusted_net_slacks = self.evaluate_net_slack()
        affected_nets = 0
        for net_id, delta in net_slack_deltas.items():
            if 0 <= net_id < len(adjusted_net_slacks) and np.isfinite(float(adjusted_net_slacks[net_id])):
                adjusted_net_slacks[net_id] = float(adjusted_net_slacks[net_id]) + float(delta)
                affected_nets += 1

        timing_hs_cpp.update_net_weights_lilith_with_net_slack(
            self.timer.raw_timer,
            self.num_nets,
            self.net_criticality if self.use_cuda else self.net_criticality_cpu,
            self.net_weights if self.use_cuda else self.net_weights_cpu,
            torch.from_numpy(self.degree_map).to(self.net_weights.device if self.use_cuda else torch.device("cpu")),
            torch.from_numpy(adjusted_net_slacks).to(self.net_weights.device if self.use_cuda else torch.device("cpu")),
            self.momentum_decay_factor,
            max_net_weight,
            self.ignore_net_degree,
            self.use_cuda,
        )

        stats["raw_setup_wns"] = raw_setup_wns
        stats["adjusted_setup_wns"] = adjusted_setup_wns
        stats["affected_nets"] = affected_nets
        stats["fallback_to_raw"] = False
        self.last_useful_skew_weighting_stats = stats
        self._sync_active_weights_to_placement()
        logging.info(
            "HeteroSTA useful-skew weighting: n=%d edges=%d margin=%.3f raw_setup_wns=%s adjusted_setup_wns=%s affected_nets=%d max_skew=%s",
            effective_n,
            graph.get("num_edges", 0),
            float(solution.get("margin")),
            "None" if raw_setup_wns is None else f"{raw_setup_wns:.3f}",
            "None" if adjusted_setup_wns is None else f"{adjusted_setup_wns:.3f}",
            affected_nets,
            self.useful_skew_max_skew,
        )
        return 0

    def evaluate_net_slack(self):
        slack = np.full(self.num_nets, np.inf, dtype=np.float32)
        flat_netpin_t = torch.from_numpy(self.flat_netpin).to(self.net_weights.device if self.use_cuda else torch.device("cpu"))
        netpin_start_t = torch.from_numpy(self.netpin_start).to(self.net_weights.device if self.use_cuda else torch.device("cpu"))
        slack_t = torch.from_numpy(slack).to(self.net_weights.device if self.use_cuda else torch.device("cpu"))
        timing_hs_cpp.evaluate_net_slack(
            self.timer.raw_timer,
            self.num_nets,
            self.num_pins,
            flat_netpin_t,
            netpin_start_t,
            slack_t,
            self.use_cuda,
        )
        if self.use_cuda:
            return slack_t.detach().cpu().numpy()
        return slack

    def _update_net_weights_raw(self, max_net_weight=np.inf, n=1):
        old_flag = self.useful_skew_weighting_flag
        self.useful_skew_weighting_flag = False
        try:
            return self.update_net_weights(max_net_weight=max_net_weight, n=n)
        finally:
            self.useful_skew_weighting_flag = old_flag
    
    def write_spef(self,file_path):
        return self.timer.raw_timer.write_spef(file_path)
    
    
    def report_wns_tns(self):
        """
        @brief report WNS and TNS in the design
        """
        return timing_hs_cpp.report_wns_tns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_wns(self):
        """
        @brief report WNS in the design
        """
        return timing_hs_cpp.report_wns(
            self.timer.raw_timer, bool(self.use_cuda))
    
    def report_tns(self):
        """
        @brief report TNS in the design
        """
        return timing_hs_cpp.report_tns(
            self.timer.raw_timer, bool(self.use_cuda))

    def update_timing(self):
        """
        This is a no-op since timing is updated in forward()
        """
        pass

    def time_unit(self):
        """
        @brief report time unit in the design
        """
        return 1e-12  # HeteroSTA uses picosecond as unit



    def dump_paths_setup_to_file(self, num_paths, nworst, file_path, use_cuda=False):
        """
        @brief dump timing paths to file using HeteroSTA
        @param num_paths number of paths to dump
        @param nworst number of worst paths per endpoint
        @param file_path output file path
        @param use_cuda whether to use CUDA
        """
        return self.timer.raw_timer.dump_paths_setup_to_file(num_paths, nworst, file_path, use_cuda)

    def report_timing_paths_by_split(self, split, n=1):
        """@brief report HeteroSTA timing paths with OpenTimer-like dictionaries."""
        return self.timer.raw_timer.report_timing_paths_by_split(split, n, self.use_cuda)

    def report_all_timing_paths_by_split(self, split):
        """@brief report all available HeteroSTA timing paths for one split."""
        return self.timer.raw_timer.report_all_timing_paths_by_split(split, self.use_cuda)

    def export_reg2reg_timing_graph(self, n=None, include_paths=False):
        """@brief export a register-to-register timing graph from HeteroSTA paths."""
        path_sets = {
            "max": self.report_timing_paths_by_split("max", n=n or 1),
            "min": self.report_timing_paths_by_split("min", n=n or 1),
        }
        return build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=include_paths)

    def export_full_reg2reg_timing_graph(self, include_paths=False):
        """@brief export a register-to-register timing graph from all available HeteroSTA paths."""
        path_sets = {
            "max": self.report_all_timing_paths_by_split("max"),
            "min": self.report_all_timing_paths_by_split("min"),
        }
        return build_reg2reg_timing_graph_from_split_paths(path_sets, include_paths=include_paths)
    
