#include "timing_hs_cpp.h"
#include "timing_hs_io_cpp.h"
#include <cstring>
#include <iostream>
#include <chrono>
#include <fstream>

#ifdef CUDA_FOUND
#include <cuda_runtime_api.h>
#endif

DREAMPLACE_BEGIN_NAMESPACE

#ifdef CUDA_FOUND
// Forward declaration of CUDA launcher
template <typename T>
void updateNetWeightCudaLauncher(
    STAHoldings* sta,
    int num_nets,
    int num_pins,
    const int* flat_netpin,
    const int* netpin_start,
    const int* pin_to_net_map,
    T* net_criticality,
    T* net_criticality_deltas,
    T* net_weights,
    T* net_weight_deltas,
    const int* degree_map,
    T momentum_decay_factor,
    T max_net_weight,
    int ignore_net_degree);

template <typename T>
void evaluateNetSlackCudaLauncher(
    STAHoldings* sta,
    int num_nets,
    int num_pins,
    const int* flat_netpin,
    const int* netpin_start,
    T* net_slack);

template <typename T>
void updateNetWeightsLilithWithNetSlackCudaLauncher(
    STAHoldings* sta,
    int num_nets,
    T* net_criticality,
    T* net_weights,
    const int* degree_map,
    const T* net_slack,
    T momentum_decay_factor,
    T max_net_weight,
    int ignore_net_degree);

namespace {

class ScopedCudaDevice {
 public:
  ScopedCudaDevice(const torch::Tensor& tensor, bool enabled)
      : active_(false), previous_device_(-1) {
    if (!enabled || !tensor.defined() || !tensor.is_cuda()) {
      return;
    }

    const int target_device = tensor.get_device();
    if (target_device < 0) {
      return;
    }

    cudaError_t status = cudaGetDevice(&previous_device_);
    if (status != cudaSuccess) {
      dreamplacePrint(kWARN, "cudaGetDevice failed before HeteroSTA call: %s\n", cudaGetErrorString(status));
      previous_device_ = -1;
    }

    if (previous_device_ == target_device) {
      return;
    }

    status = cudaSetDevice(target_device);
    if (status != cudaSuccess) {
      dreamplacePrint(
          kWARN,
          "cudaSetDevice(%d) failed before HeteroSTA call: %s\n",
          target_device,
          cudaGetErrorString(status));
      return;
    }
    active_ = true;
  }

  ~ScopedCudaDevice() {
    if (!active_ || previous_device_ < 0) {
      return;
    }
    cudaError_t status = cudaSetDevice(previous_device_);
    if (status != cudaSuccess) {
      dreamplacePrint(kWARN, "cudaSetDevice restore failed after HeteroSTA call: %s\n", cudaGetErrorString(status));
    }
  }

 private:
  bool active_;
  int previous_device_;
};

}  // namespace
#endif


/// 
/// @brief Perform timing analysis using HeteroSTA engine.
///// @param sta the HeteroSTA holdings object containing timing database.
///// @param x the horizontal coordinates of cell locations.
///// @param y the vertical coordinates of cell locations.
///// @param num_pins the number of pins in the design.
///// @param wire_resistance_per_micron unit-length resistance value.
///// @param wire_capacitance_per_micron unit-length capacitance value.
///// @param scale_factor the scaling factor to be applied to the design.
///// @param lef_unit the unit distance microns defined in the LEF file.
///// @param def_unit the unit distance microns defined in the DEF file.
///// @param slacks_rf output array for pin slacks (rise/fall).
///// @param ignore_net_degree the degree threshold for ignoring high-degree nets.
///// @param use_cuda whether to use CUDA for computation.
///
template <typename T>
int timingHeterostaCppLauncher(
		STAHoldings& sta,
		const T* x,const T* y,
		size_t num_pins,
		T wire_resistance_per_micron,
		T wire_capacitance_per_micron,
		double scale_factor, int lef_unit, int def_unit,
		float (*slacks_rf)[2],
		int ignore_net_degree, bool use_cuda){
	dreamplacePrint(kINFO, "HeteroSTA launcher started\n");

	// Convert coordinates and apply scaling for HeteroSTA units
	double unit_to_micron = scale_factor * def_unit;
	double res_unit = 1e3;  // Rust canonical units
	double cap_unit = 1e-15;
	double rf = static_cast<double>(wire_resistance_per_micron) / res_unit;
	double cf = static_cast<double>(wire_capacitance_per_micron) / cap_unit;		
	double unit_cap_xy = cf / unit_to_micron;
	double unit_res_xy = rf / unit_to_micron;

	auto beg = std::chrono::steady_clock::now();

	dreamplacePrint(kINFO,"extract rc from placement...\n");

	auto device = use_cuda ? torch::kCUDA : torch::kCPU;

	auto via_res_tensor = torch::tensor(0.0f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	auto flute_accuracy_tensor = torch::tensor(8, torch::TensorOptions().dtype(torch::kInt32).device(device));
	auto pdr_alpha_tensor = torch::tensor(0.3f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	auto use_flute_or_pdr_tensor = torch::tensor(0, torch::TensorOptions().dtype(torch::kUInt8).device(device));

	float via_res = via_res_tensor.item<float>();
	uint32_t flute_accuracy= static_cast<uint32_t>(flute_accuracy_tensor.item<int32_t>());
	float pdr_alpha = pdr_alpha_tensor.item<float>();
	uint8_t use_flute_or_pdr = use_flute_or_pdr_tensor.item<uint8_t>();	

	// If heterosta_extract_rc_from_placement panics, the process will terminate
	heterosta_extract_rc_from_placement(&sta, 
			reinterpret_cast<const float*>(x), 
			reinterpret_cast<const float*>(y),
			static_cast<float>(unit_cap_xy), 
			static_cast<float>(unit_cap_xy),
			static_cast<float>(unit_res_xy), 
			static_cast<float>(unit_res_xy),
			via_res,flute_accuracy,
			pdr_alpha,use_flute_or_pdr,	
			use_cuda);
	dreamplacePrint(kINFO,"finish rc extraction...\n");
	heterosta_update_delay(&sta, use_cuda);

	// Update arrival times and required arrival times for timing analysis
	heterosta_update_arrivals(&sta, use_cuda);

	dreamplacePrint(kINFO,"finish state updates...\n");

	// Avoid endpoint bitmap probing here as well. The current HeteroSTA
	// library can panic in heterosta_get_is_endpoint on large ICCAD2015
	// designs, while the timing update path itself does not require this
	// debug-side endpoint scan.
	//heterosta_launch_debug_shell(&sta);

	// Report pin slacks
	if(!use_cuda && slacks_rf!=nullptr)
	{
		heterosta_report_slacks_at_max(&sta, slacks_rf, use_cuda);
		dreamplacePrint(kDEBUG, "HeteroSTA slack report completed\n");
		dreamplacePrint(kDEBUG, "Skipping endpoint-wise slack dump because endpoint probing is disabled.\n");
	}
	auto end = std::chrono::steady_clock::now();
	auto usc = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
	dreamplacePrint(kINFO, "finish state updates (%f s)\n", usc.count() * 0.001);
	return 0;
}

// Implementation of the forward method
void TimingHeterostaCpp::forward(
		STAHoldings& sta, torch::Tensor pos,
		size_t num_pins,
		double wire_resistance_per_micron,
		double wire_capacitance_per_micron,
		double scale_factor, int lef_unit, int def_unit,
		torch::Tensor slacks_rf,
		int ignore_net_degree, bool use_cuda) {
	// Check tensor properties for input validation
	CHECK_EVEN(pos);
	CHECK_CONTIGUOUS(pos);
	// Device consistency check
	bool pos_is_cuda = pos.is_cuda();

	if (use_cuda != pos_is_cuda) {
		dreamplacePrint(kWARN, "Device mismatch detected but proceeding (pos_pin should handle device placement)\n");
	}
	// Check slack tensor if provided
	float (*slacks_rf_ptr)[2] = nullptr;
	if (slacks_rf.numel() > 0) {
		CHECK_CONTIGUOUS(slacks_rf);
		float* slacks_rf_data = slacks_rf.data_ptr<float>();
		slacks_rf_ptr = reinterpret_cast<float (*)[2]>(slacks_rf_data);
	}

#ifdef CUDA_FOUND
	if (use_cuda && pos.is_cuda()) {
		auto target_device = static_cast<uint8_t>(pos.get_device());
		heterosta_set_cuda_device_id(&sta, target_device);
		dreamplacePrint(kDEBUG, "Set HeteroSTA CUDA device to %u before timing forward\n", target_device);
	}
	ScopedCudaDevice device_guard(pos, use_cuda);
#endif

	DREAMPLACE_DISPATCH_FLOATING_TYPES(
			pos, "timingHeterostaCppLauncher",
			[&] {
			timingHeterostaCppLauncher<scalar_t>(
					sta,
					DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
					num_pins,
					wire_resistance_per_micron,
					wire_capacitance_per_micron,
					scale_factor, lef_unit, def_unit,
					slacks_rf_ptr,
					ignore_net_degree,
					use_cuda
					);
			});
}

///
/// @brief Update net weights based on timing criticality using HeteroSTA (template launcher).
/// @param sta the HeteroSTA holdings object containing timing database.
/// @param n the maximum number of critical paths to analyze.
/// @param num_nets the number of nets in the design.
/// @param num_pins the number of pins in the design.
/// @param flat_netpin the flattened netpin array.
/// @param netpin_start the starting indices for each net.
/// @param pin_to_net_map the pin to net mapping array.
/// @param net_criticality the criticality values of nets (array).
/// @param net_criticality_deltas the criticality delta values of nets (array).
/// @param net_weights the weights of nets (array).
/// @param net_weight_deltas the increment of net weights.
/// @param degree_map the degree map of nets.
/// @param max_net_weight the maximum net weight in timing optimization.
/// @param momentum_decay_factor the decay factor in momentum iteration.
/// @param ignore_net_degree the net degree threshold for ignoring high-degree nets.
/// @param num_threads number of threads for parallel computing.
///
template <typename T>
void updateNetWeightCppLauncher(
		STAHoldings& sta,
		int num_nets, 
		int num_pins,
		const int* flat_netpin, 
		const int* netpin_start,
		const int* pin_to_net_map,
		T* net_criticality, 
		T* net_criticality_deltas,
		T* net_weights, 
		T* net_weight_deltas,
		const int* degree_map,
		T momentum_decay_factor, 
		T max_net_weight,
		int ignore_net_degree, 
		int num_threads) {

	// Get WNS/TNS from HeteroSTA
	float wns, tns;
	bool success = heterosta_report_wns_tns_max(&sta, &wns, &tns, false);

	// Get pin slacks from HeteroSTA
	static std::vector<float> slack_data;
	if (slack_data.size() < num_pins * 2) {
		slack_data.resize(num_pins * 2);
	}
	float (*slack_array)[2] = reinterpret_cast<float(*)[2]>(slack_data.data());
	heterosta_report_slacks_at_max(&sta, slack_array, false);

	// Apply net weighting using momentum-based criticality update
	if(!(wns < 0)) wns = 0;

#pragma omp parallel for num_threads(num_threads)
	for(int net_i = 0; net_i < num_nets; ++net_i) {

		float net_slack = std::numeric_limits<float>::max();
		int np_s = netpin_start[net_i], np_e = netpin_start[net_i + 1];

		// Calculate net slack as minimum of all pin slacks in the net
		for(int np_i = np_s; np_i < np_e; ++np_i) {
			int pin_i = flat_netpin[np_i];
			if (pin_i < num_pins) {
				// Take the worst (minimum) of rise/fall slack for this pin
				float pin_worst_slack = std::min(slack_array[pin_i][0], 
						slack_array[pin_i][1]);
				net_slack = std::min(net_slack, pin_worst_slack);
			}
		}


		if(wns < 0) {
			// Calculate normalized criticality
			float nc = (net_slack < 0) ? std::max(0.f, net_slack / wns) : 0;
			// Apply momentum-based criticality update 
			net_criticality[net_i] = std::pow(1 + net_criticality[net_i], momentum_decay_factor) * 
				std::pow(1 + nc, 1 - momentum_decay_factor) - 1;
		}

		if(degree_map[net_i]>ignore_net_degree) continue;
		net_weights[net_i] *= (1 + net_criticality[net_i]);

		if(net_weights[net_i] > max_net_weight)        net_weights[net_i] = max_net_weight; 
	}

}

// Implementation of the update_net_weights method
void TimingHeterostaCpp::update_net_weights(
		STAHoldings& sta, int n,
		int num_nets,
		int num_pins,
		torch::Tensor flat_netpin, torch::Tensor netpin_start,
		torch::Tensor pin2net_map,
		torch::Tensor net_criticality, torch::Tensor net_criticality_deltas,
		torch::Tensor net_weights, torch::Tensor net_weight_deltas,
		torch::Tensor degree_map,
		double max_net_weight,
		double momentum_decay_factor,
		int ignore_net_degree, bool use_cuda) {

	// Check tensor properties
	CHECK_CONTIGUOUS(flat_netpin);
	CHECK_CONTIGUOUS(netpin_start);
	CHECK_CONTIGUOUS(pin2net_map);
	CHECK_CONTIGUOUS(net_criticality);
	CHECK_CONTIGUOUS(net_weights);
	CHECK_CONTIGUOUS(net_weight_deltas);
	CHECK_CONTIGUOUS(degree_map);

#ifdef CUDA_FOUND
	if (use_cuda && net_weights.is_cuda()) {
		auto target_device = static_cast<uint8_t>(net_weights.get_device());
		heterosta_set_cuda_device_id(&sta, target_device);
		dreamplacePrint(kDEBUG, "Set HeteroSTA CUDA device to %u before net weight update\n", target_device);
	}
	ScopedCudaDevice device_guard(net_weights, use_cuda);
#endif

	if (use_cuda) {
#ifdef CUDA_FOUND
        DREAMPLACE_DISPATCH_FLOATING_TYPES(
            net_criticality, "updateNetWeightCudaLauncher",
            [&] {
                updateNetWeightCudaLauncher<scalar_t>(
                    &sta,
                    num_nets,
                    num_pins,
                    DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
                    DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
                    DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
                    DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(net_criticality_deltas, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(net_weight_deltas, scalar_t),
                    DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
                    static_cast<scalar_t>(momentum_decay_factor),
                    static_cast<scalar_t>(max_net_weight),
                    ignore_net_degree);
            });
#else
        dreamplacePrint(kERROR, "CUDA not available but use_cuda=true\n");
#endif
    } else {
	DREAMPLACE_DISPATCH_FLOATING_TYPES(
			net_criticality, "updateNetWeightCppLauncher",
			[&] {
			updateNetWeightCppLauncher<scalar_t>(
					sta,
					num_nets,
					num_pins,
					DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
					DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
					DREAMPLACE_TENSOR_DATA_PTR(pin2net_map, int),
					DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_criticality_deltas, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_weight_deltas, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
					static_cast<scalar_t>(momentum_decay_factor),
					static_cast<scalar_t>(max_net_weight),
					ignore_net_degree,
					at::get_num_threads());
			});
	}

}

template <typename T>
void evaluateNetSlackCppLauncher(
		STAHoldings& sta,
		int num_nets,
		int num_pins,
		const int* flat_netpin,
		const int* netpin_start,
		T* net_slack) {
	static std::vector<float> slack_data;
	if (slack_data.size() < num_pins * 2) {
		slack_data.resize(num_pins * 2);
	}
	float (*slack_array)[2] = reinterpret_cast<float(*)[2]>(slack_data.data());
	heterosta_report_slacks_at_max(&sta, slack_array, false);

	for (int net_i = 0; net_i < num_nets; ++net_i) {
		float value = std::numeric_limits<float>::max();
		for (int np_i = netpin_start[net_i]; np_i < netpin_start[net_i + 1]; ++np_i) {
			int pin_i = flat_netpin[np_i];
			if (pin_i < 0 || pin_i >= num_pins) {
				continue;
			}
			float pin_worst_slack = std::min(slack_array[pin_i][0], slack_array[pin_i][1]);
			value = std::min(value, pin_worst_slack);
		}
		net_slack[net_i] = static_cast<T>(value);
	}
}

void TimingHeterostaCpp::evaluate_net_slack(
		STAHoldings& sta,
		int num_nets,
		int num_pins,
		torch::Tensor flat_netpin,
		torch::Tensor netpin_start,
		torch::Tensor slack,
		bool use_cuda) {
	CHECK_CONTIGUOUS(flat_netpin);
	CHECK_CONTIGUOUS(netpin_start);
	CHECK_CONTIGUOUS(slack);

	if (use_cuda) {
		DREAMPLACE_DISPATCH_FLOATING_TYPES(
				slack, "evaluateNetSlackCudaLauncher",
				[&] {
				evaluateNetSlackCudaLauncher<scalar_t>(
						&sta,
						num_nets,
						num_pins,
						DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
						DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
						DREAMPLACE_TENSOR_DATA_PTR(slack, scalar_t));
				});
		return;
	}

	DREAMPLACE_DISPATCH_FLOATING_TYPES(
			slack, "evaluateNetSlackCppLauncher",
			[&] {
			evaluateNetSlackCppLauncher<scalar_t>(
					sta,
					num_nets,
					num_pins,
					DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
					DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
					DREAMPLACE_TENSOR_DATA_PTR(slack, scalar_t));
			});
}

template <typename T>
void updateNetWeightsLilithWithNetSlackCppLauncher(
		STAHoldings& sta,
		int num_nets,
		T* net_criticality,
		T* net_weights,
		const int* degree_map,
		const T* net_slack,
		T momentum_decay_factor,
		T max_net_weight,
		int ignore_net_degree,
		bool use_cuda) {
	float wns = 0.0f;
	float tns = 0.0f;
	bool success = heterosta_report_wns_tns_max(&sta, &wns, &tns, use_cuda);
	if (!success) {
		dreamplacePrint(kWARN, "report_wns_tns_max failed; skip lilith-with-net-slack update\n");
		return;
	}

	for (int net_i = 0; net_i < num_nets; ++net_i) {
		float slack = static_cast<float>(net_slack[net_i]);
		if (wns < 0) {
			float nc = (slack < 0) ? std::max(0.f, slack / wns) : 0.f;
			net_criticality[net_i] = std::pow(1 + net_criticality[net_i], momentum_decay_factor) *
				std::pow(1 + nc, 1 - momentum_decay_factor) - 1;
		}
		if (degree_map[net_i] > ignore_net_degree) continue;
		net_weights[net_i] *= (1 + net_criticality[net_i]);
		if (net_weights[net_i] > max_net_weight) {
			net_weights[net_i] = max_net_weight;
		}
	}
}

void TimingHeterostaCpp::update_net_weights_lilith_with_net_slack(
		STAHoldings& sta,
		int num_nets,
		torch::Tensor net_criticality,
		torch::Tensor net_weights,
		torch::Tensor degree_map,
		torch::Tensor net_slack,
		double momentum_decay_factor,
		double max_net_weight,
		int ignore_net_degree,
		bool use_cuda) {
	CHECK_CONTIGUOUS(net_criticality);
	CHECK_CONTIGUOUS(net_weights);
	CHECK_CONTIGUOUS(degree_map);
	CHECK_CONTIGUOUS(net_slack);

	if (use_cuda) {
		DREAMPLACE_DISPATCH_FLOATING_TYPES(
				net_weights, "updateNetWeightsLilithWithNetSlackCudaLauncher",
				[&] {
				updateNetWeightsLilithWithNetSlackCudaLauncher<scalar_t>(
						&sta,
						num_nets,
						DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
						DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
						DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
						DREAMPLACE_TENSOR_DATA_PTR(net_slack, scalar_t),
						static_cast<scalar_t>(momentum_decay_factor),
						static_cast<scalar_t>(max_net_weight),
						ignore_net_degree);
				});
		return;
	}

	DREAMPLACE_DISPATCH_FLOATING_TYPES(
			net_weights, "updateNetWeightsLilithWithNetSlackCppLauncher",
			[&] {
			updateNetWeightsLilithWithNetSlackCppLauncher<scalar_t>(
					sta,
					num_nets,
					DREAMPLACE_TENSOR_DATA_PTR(net_criticality, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t),
					DREAMPLACE_TENSOR_DATA_PTR(degree_map, int),
					DREAMPLACE_TENSOR_DATA_PTR(net_slack, scalar_t),
					static_cast<scalar_t>(momentum_decay_factor),
					static_cast<scalar_t>(max_net_weight),
					ignore_net_degree,
					use_cuda);
			});
}

DREAMPLACE_END_NAMESPACE
