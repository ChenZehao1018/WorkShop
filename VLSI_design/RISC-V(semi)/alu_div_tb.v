// Company           :   tud                      
// Author            :   chze22            
// E-Mail            :   <email>                    
//                    			
// Filename          :   tb_alu_div.v                
// Project Name      :   prz    
// Subproject Name   :   gf28_template    
// Description       :   <short description>            
//
// Create Date       :   Mon Dec  5 13:57:31 2022 
// Last Change       :   $Date$
// by                :   $Author$                  			
//------------------------------------------------------------
`timescale 1ns/10ps
module tb_alu_div (
);

 
  reg  clk ;
  reg  nReset ;
  reg  alu_div_stb_i ;
  reg  [ 4 : 0 ] alu_div_funct_i ;
  reg  [ 31 : 0 ] alu_div_op1_i ;
  reg  [ 31 : 0 ] alu_div_op2_i ;
  
  wire [ 31 : 0 ] alu_div_res_o ;
  wire alu_div_done_o ;
 
  alu_div alu_div_i (
		.clk (clk ),
		.nReset (nReset ),
		.alu_div_stb_i (alu_div_stb_i ),
		.alu_div_funct_i (alu_div_funct_i ),
		.alu_div_op1_i (alu_div_op1_i ),
		.alu_div_op2_i (alu_div_op2_i ),
		.alu_div_res_o (alu_div_res_o ),
		.alu_div_done_o (alu_div_done_o )
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
      alu_div_stb_i = 1'b1;
      alu_div_funct_i = 5'b10010;
      alu_div_op1_i = 32'b0101110011010;
      alu_div_op2_i = 32'b01010;
      #(CLKPERIODE * 1)
      alu_div_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      alu_div_stb_i = 1'b1;
      alu_div_funct_i = 5'b10100;
      alu_div_op1_i = 32'b0100101001;
      alu_div_op2_i = 32'b010000;
      #(CLKPERIODE * 1)
      alu_div_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      alu_div_stb_i = 1'b1;
      alu_div_funct_i = 5'b10010;
      alu_div_op1_i = -32'b0101110011010;
      alu_div_op2_i = 32'b01010;
      #(CLKPERIODE * 1)
      alu_div_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      alu_div_stb_i = 1'b1;
      alu_div_funct_i = 5'b10100;
      alu_div_op1_i = -32'b0100101001;
      alu_div_op2_i = 32'b010000;
      #(CLKPERIODE * 1)
      alu_div_stb_i = 1'b0;
      #(CLKPERIODE * 40)
      $finish();
  end


endmodule
