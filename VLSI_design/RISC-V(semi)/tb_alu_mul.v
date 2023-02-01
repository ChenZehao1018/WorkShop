`timescale 1ns/10ps
module tb_alu_mul (
);

 
  reg  clk ;
  reg  nReset ;
  reg  alu_mul_stb_i ;
  reg  [ 4 : 0 ] alu_mul_funct_i ;
  reg  [ 31 : 0 ] alu_mul_op1_i ;
  reg  [ 31 : 0 ] alu_mul_op2_i ;
  
  wire [ 31 : 0 ] alu_mul_res_o ;
  wire alu_mul_done_o ;
 
  alu_mul alu_mul_i (
		.clk (clk ),
		.nReset (nReset ),
		.alu_mul_stb_i (alu_mul_stb_i ),
		.alu_mul_funct_i (alu_mul_funct_i ),
		.alu_mul_op1_i (alu_mul_op1_i ),
		.alu_mul_op2_i (alu_mul_op2_i ),
		.alu_mul_res_o (alu_mul_res_o ),
		.alu_mul_done_o (alu_mul_done_o )
        );

  ///////////////////////////////////////////////
  //// Template for clk and reset generation ////
  //// uncomment it when needed              ////
  ///////////////////////////////////////////////
  parameter CLKPERIODE = 100;

  initial clk = 1'b1;
  always #(CLKPERIODE/2.0) clk = !clk;

  initial begin
      nReset = 1'b0;
      #33
      nReset = 1'b1;
      #33
      alu_mul_stb_i = 1'b1;
      alu_mul_funct_i = 5'b01110;
      alu_mul_op1_i = -32'b011100011111;
      alu_mul_op2_i = 32'b01010;
      #(CLKPERIODE * 1)
      alu_mul_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      alu_mul_stb_i = 1'b1;
      alu_mul_funct_i = 5'b01111;
      alu_mul_op1_i = -32'b0100101001;
      alu_mul_op2_i = 32'b010000;
      #(CLKPERIODE * 1)
      alu_mul_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      alu_mul_stb_i = 1'b1;
      alu_mul_funct_i = 5'b10000;
      alu_mul_op1_i = -32'b0101110011010;
      alu_mul_op2_i = 32'b0101111111110;
      #(CLKPERIODE * 1)
      alu_mul_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      alu_mul_stb_i = 1'b1;
      alu_mul_funct_i = 5'b10001;
      alu_mul_op1_i = -32'b0100101001;
      alu_mul_op2_i = 32'b011111110111000;
      #(CLKPERIODE * 1)
      alu_mul_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      $finish();
  end


endmodule
