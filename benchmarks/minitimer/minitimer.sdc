create_clock -period 180 -name clk [get_ports clk]

set_input_delay 0 -min -rise [get_ports clk] -clock clk
set_input_delay 0 -min -fall [get_ports clk] -clock clk
set_input_delay 0 -max -rise [get_ports clk] -clock clk
set_input_delay 0 -max -fall [get_ports clk] -clock clk

# [Jsy] The original single-register input path dominated hold, so the
# reg-to-reg hold example never surfaced. Push the PI arrival later so the
# worst min path comes from reg0 -> reg1 instead of in1 -> reg0.
set_input_delay 60 -min -rise [get_ports in1] -clock clk
set_input_delay 60 -min -fall [get_ports in1] -clock clk
set_input_delay 60 -max -rise [get_ports in1] -clock clk
set_input_delay 60 -max -fall [get_ports in1] -clock clk

set_input_transition 5 -min -rise [get_ports clk] -clock clk
set_input_transition 5 -min -fall [get_ports clk] -clock clk
set_input_transition 5 -max -rise [get_ports clk] -clock clk
set_input_transition 5 -max -fall [get_ports clk] -clock clk

set_input_transition 5 -min -rise [get_ports in1]
set_input_transition 5 -min -fall [get_ports in1]
set_input_transition 5 -max -rise [get_ports in1]
set_input_transition 5 -max -fall [get_ports in1]

set_load -pin_load 4 [get_ports out1]
