module minitimer (clk, in1, out1);

input clk;
input in1;
output out1;

wire n_q0;
wire n_mid;

DFF_X1 reg0 ( .D(in1), .CK(clk), .Q(n_q0), .QN() );
INV_X1 comb0 ( .A(n_q0), .ZN(n_mid) );
DFF_X1 reg1 ( .D(n_mid), .CK(clk), .Q(out1), .QN() );

endmodule
