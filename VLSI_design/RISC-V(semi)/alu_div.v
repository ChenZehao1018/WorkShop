module alu_div(
    input wire clk,
    input wire nReset,
    input wire alu_div_stb_i,
//from ex
    input wire [4:0] alu_div_funct_i,
    input wire [31:0] alu_div_op1_i,
    input wire [31:0] alu_div_op2_i,
//to ex
    output wire [31:0] alu_div_res_o,
    output wire alu_div_done_o
);

reg div_done_r, div_done_next;
reg [3:0] state_r, state_next;
reg [5:0] cnt_r, cnt_next;
reg[31:0] alu_div_res_r, alu_div_res_next;
reg[63:0] dividend_r, dividend_next;
reg[63:0] divisor_r, divisor_next;
reg is_neg_r, is_neg_next;

localparam IDLE = 4'b0001;
localparam INI = 4'b0010;
localparam RUN = 4'b0100;
localparam DONE = 4'b1000;

assign alu_div_res_o = alu_div_res_r;
assign alu_div_done_o = div_done_r;

always @ (posedge clk or negedge nReset) begin
    if(nReset == 1'b0)begin
        state_r <= IDLE;
        div_done_r <= 1'b0;
        cnt_r <= 6'b0;
        alu_div_res_r <= 32'b0;
        dividend_r <= 64'b0;
        divisor_r <= 64'b0;
        is_neg_r <= 1'b0;
    end
    else begin
        state_r <= state_next;
        div_done_r <= div_done_next;
        alu_div_res_r <= alu_div_res_next;
        cnt_r <= cnt_next;
        dividend_r <= dividend_next;
        divisor_r <= divisor_next;
        is_neg_r <= is_neg_next;
    end
end
always @ ( * ) begin
    state_next <= state_r;
    div_done_next <= div_done_r;
    alu_div_res_next <= alu_div_res_r;
    cnt_next <= cnt_r;
    dividend_next <= dividend_r;
    divisor_next <= divisor_r;
    is_neg_next <= is_neg_r;
    case(state_r)
    IDLE:begin
        div_done_next <= 1'b0;
        alu_div_res_next <= 32'b0;
        cnt_next <= 32'b0;
        dividend_next <= 64'b0;
        divisor_next <= 64'b0;
        is_neg_next <= 1'b0;
        
        if(alu_div_stb_i) begin
        state_next <= INI;
        end
    end
    INI:begin
        if((alu_div_funct_i == 5'b10010) | (alu_div_funct_i == 5'b10100))begin
            dividend_next <= alu_div_op1_i[31]?{{32{1'b0}},-alu_div_op1_i}:{{32{1'b0}},alu_div_op1_i};
            divisor_next <= alu_div_op2_i[31]?{-alu_div_op2_i,{32{1'b0}}}:{alu_div_op2_i,{32{1'b0}}};
        end
        else if((alu_div_funct_i == 5'b10011)|(alu_div_funct_i == 5'b10101))begin
            dividend_next <= {{32{1'b0}},alu_div_op1_i};
            divisor_next <= {alu_div_op2_i,{32{1'b0}}};
        end
        state_next <= RUN;
        cnt_next <= 6'd32;
        is_neg_next <= alu_div_op1_i[31] ^ alu_div_op2_i[31];
    end
    RUN:begin
        if(dividend_r == 64'b0)begin
        state_next <= DONE;
        end
        else begin
            if(cnt_r == 6'b0)begin
            dividend_next <= (dividend_r < divisor_r)? dividend_r:(dividend_r - divisor_r + 1'b1);
            state_next <= DONE;
            end
            else begin
            dividend_next <= (dividend_r < divisor_r)? dividend_r << 1'b1:(dividend_r - divisor_r + 1'b1) << 1'b1;
            state_next <= RUN;
            cnt_next <= cnt_r - 1'b1;
            end
        end
    end
    DONE:begin
        if(alu_div_funct_i == 5'b10010)begin
            alu_div_res_next <= is_neg_r? (-dividend_r[31:0]):(dividend_r[31:0]);
        end
        else if(alu_div_funct_i == 5'b10011)begin
            alu_div_res_next <= dividend_r[31:0];
        end
        else if(alu_div_funct_i == 5'b10100)begin
            alu_div_res_next <= is_neg_r? (-dividend_r[63:32]):(dividend_r[63:32]);
        end
        else if(alu_div_funct_i == 5'b10101)begin
            alu_div_res_next <= dividend_r[63:32];
        end
        div_done_next <= 1'b1;
        state_next <= IDLE;
    end
    endcase
end


endmodule
