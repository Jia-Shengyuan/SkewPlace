##
# @file   Placer.py
# @author Enzo
# @date   Apr 2021
# @brief  The timer required in the timing-driven mode
#

import time
import logging

class Timer(object):
    """
    @brief The timer python class. The timer supports both Opentimer and HeteroSTA engines.
    """
    def __init__(self,timer_engine = "heterosta"):
        """The initialization of the timer python class.
        In fact we do not prefer using timer directly in python, even in
        the timing-driven mode. Ideally, everything should smoothly goes
        on at the underlying level.
        """
        self.raw_timer = None
        self.timer_engine = timer_engine.lower()

        # Validate engine selection
        if self.timer_engine not in ["opentimer","heterosta"]:
            raise ValueError(f"Unsupported timer engine:{timer_engine}.Must be 'opentimer' or 'heterosta'")
        
        # Lazy import based on engine selection
        if self.timer_engine == "opentimer":
            try:
                import dreamplace.ops.timing.timing as timing
                import dreamplace.ops.timing.timing_cpp as timing_cpp
                self.timing_module = timing
                self.timing_cpp_module = timing_cpp
                logging.info("Using OpenTimer timing analysis engine")
            except ImportError as e:
                raise ImportError(f"Failed to import OpenTimer modules: {e}")
        else:  # heterosta
            try:
                import dreamplace.ops.timing_heterosta.timing_hs as timing
                import dreamplace.ops.timing_heterosta.timing_heterosta_cpp as timing_cpp
                self.timing_module = timing
                self.timing_cpp_module = timing_cpp
                logging.info("Using HeteroSTA timing analysis engine")
            except ImportError as e:
                raise ImportError(f"Failed to import HeteroSTA modules: {e}")

    def read(self, params,placedb=None): 
        """
        @brief read using c++ and save the pybind object.
        @param params the parameters specified in json.
        @param placedb the placement database (required for HeteroSTA)
        """
        if self.timer_engine == "opentimer":
            self.raw_timer = self.timing_module.TimingIO.read(params)
        else:  # heterosta
            if placedb is None:
                raise ValueError("HeteroSTA requires placedb parameter")
            self.raw_timer = self.timing_module.TimingIO.read(params, placedb)
    
    def __call__(self, params, placedb):
        """
        @brief top API to read placement files 
        @param params parameters
        @param placedb the placement database.
        """
        tt = time.time()
        self.read(params,placedb)
        self.placedb = placedb
        logging.info("reading timer constraints takes %g seconds" % \
            (time.time() - tt))

    def update_timing(self):
        """@brief update timing.
        Note that the parsers only build tasks in cpp-taskflow and will
        not be executed before the update_timing call.
        """        
        if self.timer_engine == "opentimer":
            return self.raw_timer.update_timing()
        else: # heterosta
            pass
    
    # ---------------------
    # In the following,we keep original methods for opentimer.
    # Not implemented in the heterosta mode
    
    def report_timing(self, n):
        """@brief report timing paths and related nets.
        The underlying implementation is completed in the c++ core.
        The return value is in fact a list of net indices. The net
        indices can be very useful when aiming at adjusting the net
        weight of specific nets.
        """
        # The extraction should be called after the timing update every
        # time. The two functions are separated.
        if self.timer_engine != "opentimer":
            raise NotImplementedError("report_timing is only implemented for OpenTimer")
        # [Jsy] This used to call the global OpenTimer binding directly. Route
        # it through the module selected at Timer construction so the wrapper
        # stays consistent with the active timing backend.
        return self.timing_cpp_module.report_timing(
            self.raw_timer, n, self.placedb.net_name2id_map)

    def report_timing_paths(self, n=1):
        """@brief report detailed timing path information."""
        # [Jsy] Originally Timer only exposed net ids on critical paths. Add a
        # path-level API here so post-placement reg-to-reg extraction can reuse
        # the existing timer instance without changing placement flow.
        if self.timer_engine != "opentimer":
            raise NotImplementedError("report_timing_paths is only implemented for OpenTimer")
        return self.timing_cpp_module.report_timing_paths(self.raw_timer, n)

    def report_timing_paths_by_split(self, split, n=1):
        """@brief report detailed timing path information for a single split."""
        if self.timer_engine != "opentimer":
            raise NotImplementedError("report_timing_paths_by_split is only implemented for OpenTimer")
        return self.timing_cpp_module.report_timing_paths_by_split(self.raw_timer, n, split)

    def report_test_paths_by_split(self, split, n=None):
        """@brief report detailed sequential test paths for a single split."""
        if self.timer_engine != "opentimer":
            raise NotImplementedError("report_test_paths_by_split is only implemented for OpenTimer")
        # [Jsy] Exporting every sequential test path is too expensive on large
        # benchmarks. Allow callers to cap the OpenTimer report count and then
        # filter the returned paths to sequential endpoints in Python.
        if n is not None:
            paths = [
                path for path in self.report_timing_paths_by_split(split, n=n)
                if path.get("endpoint_type") == "test"
            ]
        else:
            paths = self.timing_cpp_module.report_test_paths_by_split(self.raw_timer, split)
        # [Jsy] OpenTimer can return an empty list for K>1 on tiny examples even
        # when the top-1 split path exists. Fall back to K=1 so the Python-level
        # reg-to-reg export still sees the sequential path we need for prototyping.
        if not paths and self.raw_timer.num_tests() > 0:
            paths = [
                path for path in self.report_timing_paths_by_split(split, n=1)
                if path.get("endpoint_type") == "test"
            ]
        return paths

    def export_reg2reg_timing_graph(self, n=None, include_paths=False):
        """@brief export a register-to-register timing graph from OpenTimer paths."""
        # [Jsy] Keep the graph export on Timer instead of a separate entry
        # point so downstream prototypes can work from the same STA state.
        if self.timer_engine != "opentimer":
            raise NotImplementedError("reg-to-reg timing graph export is only implemented for OpenTimer")
        return self.timing_module.export_reg2reg_timing_graph(
            self, n=n, include_paths=include_paths)

    def solve_useful_skew(self, n=None, max_skew=None):
        """@brief solve the minimal useful-skew LP on exported timing paths."""
        # [Jsy] The original Timer stopped at reporting timing. Expose the
        # minimal LP prototype here to validate useful-skew after placement
        # before touching the main placement objective.
        if self.timer_engine != "opentimer":
            raise NotImplementedError("useful skew prototype is only implemented for OpenTimer")
        return self.timing_module.solve_useful_skew_from_timer(
            self, n=n, max_skew=max_skew)

    # ----------------------    
    # In the following, we define some accessors. Note that they are
    # not properties so call them in a function way.
    # All these accessors directly call the instance variable of raw
    # timer binded from c++ operators inside their implementations.
    def num_primary_inputs(self):
        """@brief aquire the number of primary inputs in the design.
        """
        return self.raw_timer.num_primary_inputs()

    def num_primary_outputs(self):
        """@brief aquire the number of primary outputs in the design.
        """
        return self.raw_timer.num_primary_outputs()

    def num_nets(self):
        """@brief aquire the total number of nets in the design.
        """
        return self.raw_timer.num_nets()

    def num_pins(self):
        """@brief aquire the total number of pins in the design.
        """
        return self.raw_timer.num_pins()

    def num_arcs(self):
        """@brief aquire the total number of arcs in the design.
        """
        return self.raw_timer.num_arcs()

    def num_gates(self):
        """@brief aquire the total number of gates in the design.
        """
        return self.raw_timer.num_gates()

    def num_tests(self):
        """@brief aquire the total number of tests in the design.
        """
        return self.raw_timer.num_tests()

    def num_sccs(self):
        """@brief aquire the total number of sccs in the design.
        """
        return self.raw_timer.num_sccs()

    def num_worst_endpoints(self):
        """@brief aquire the maximum size of end points for EL/RF
        """
        return self.raw_timer.num_worst_endpoints()

    # ----------------------    
    # In the following, we define some dump functions.
    def dump_graph(self, fout=None):
        """@brief dump timing graph
        """
        if fout is None: return self.raw_timer.dump_graph()
        return self.raw_timer.dump_graph_file(fout)

    def dump_taskflow(self, fout=None):
        """@brief dump taskflow graph
        """
        if fout is None: return self.raw_timer.dump_taskflow()
        return self.raw_timer.dump_taskflow_file(fout)

    def dump_netload(self, fout=None):
        """@brief dump netload
        """
        if fout is None: return self.raw_timer.dump_netload()
        return self.raw_timer.dump_netload_file(fout)

    def dump_pin_cap(self, fout=None):
        """@brief dump pin capacitance values
        """
        if fout is None: return self.raw_timer.dump_pin_cap()
        return self.raw_timer.dump_pin_cap_file(fout)

    def dump_at(self, fout=None):
        """@brief dump at
        """
        if fout is None: return self.raw_timer.dump_at()
        return self.raw_timer.dump_at_file(fout)

    def dump_rat(self, fout=None):
        """@brief dump rat
        """
        if fout is None: return self.raw_timer.dump_rat()
        return self.raw_timer.dump_rat_file(fout)

    def dump_slew(self, fout=None):
        """@brief dump slews
        """
        if fout is None: return self.raw_timer.dump_slew()
        return self.raw_timer.dump_slew_file(fout)

    def dump_slack(self, fout=None):
        """@brief dump slacks
        """
        if fout is None: return self.raw_timer.dump_slack()
        return self.raw_timer.dump_slack_file(fout)

    def dump_timer(self, fout=None):
        """@brief dump the current timer
        """
        if fout is None: return self.raw_timer.dump_timer()
        return self.raw_timer.dump_timer_file(fout)

    def dump_spef(self, fout=None):
        """@brief dump slews
        """
        if fout is None: return self.raw_timer.dump_spef()
        return self.raw_timer.dump_spef_file(fout)

    def dump_rctree(self, fout=None):
        """@brief dump slews
        """
        if fout is None: return self.raw_timer.dump_rctree()
        return self.raw_timer.dump_rctree_file(fout)
    
    # ----------------------    
    # In the following, we define some report functions.
    def report_tns(self, split=None, tran=None):
        """@brief report tns value
        """
        if split is None and tran is None:
            return self.raw_timer.report_tns_all()
        elif split is not None and tran is None:
            return self.raw_timer.report_tns_el(split)
        elif split is None and tran is not None:
            return self.raw_timer.report_tns_rl(tran)
        else:
            return self.raw_timer.report_tns_el_rf(split, tran)
    
    def report_wns(self, split=None, tran=None):
        """@brief report wns value
        """
        if split is None and tran is None:
            return self.raw_timer.report_wns_all()
        elif split is not None and tran is None:
            return self.raw_timer.report_wns_el(split)
        elif split is None and tran is not None:
            return self.raw_timer.report_wns_rl(tran)
        else:
            return self.raw_timer.report_wns_el_rf(split, tran)

    def report_tns_elw(self, split=None):
        """@brief report tns value with only the worst between rise
        and fall considered
        """
        if split is None:
            return self.raw_timer.report_tns_elw()
        else:
            return self.raw_timer.report_tns_elw(split)

    # ----------------------
    # In the following, we define some functions related to units.
    def cap_unit(self):
        """
        @brief return the capacitance unit value (farah)
        @return the cap unit value (farah)
        """
        return self.raw_timer.cap_unit()
    
    def res_unit(self):
        """
        @brief return the resistance unit value (ohm)
        @return the res unit value (ohm)
        """
        return self.raw_timer.res_unit()

    def time_unit(self):
        """
        @brief return the current time unit value (s)
        @return the time unit value (s)
        """
        return self.raw_timer.time_unit()
