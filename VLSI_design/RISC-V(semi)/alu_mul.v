module alu_mul (
    input wire clk,
    input wire nReset,
    input wire alu_mul_stb_i,
    //from ex
    input wire [4:0] alu_mul_funct_i,
    input wire [31:0] alu_mul_op1_i,
    input wire [31:0] alu_mul_op2_i,
    //to ex
    output wire alu_mul_done_o,
    output wire [31:0] alu_mul_res_o
);
reg is_neg_r ,is_neg_next;
reg mul_done_r, mul_done_next;
reg[3:0] state_r, state_next;
reg[31:0] alu_mul_res_r, alu_mul_res_next;
reg[31:0] op2_r, op2_next;
reg[63:0] op1_r, op1_next;
reg[63:0] acc_r, acc_next;


localparam IDLE = 4'b0001;
localparam INI = 4'b0010;
localparam RUN = 4'b0100;
localparam DONE = 4'b1000;

assign alu_mul_res_o = alu_mul_res_r;
assign alu_mul_done_o = mul_done_r;


always @ (posedge clk or negedge nReset) begin
    if(nReset == 1'b0)begin
        is_neg_r <= 1'b0;
        mul_done_r <= 1'b0;
        state_r <= IDLE;
        alu_mul_res_r <= 32'b0;
        op2_r <= 32'b0;
        op1_r <= 64'b0;
        acc_r <= 64'b0;
    end
    else begin
        is_neg_r <= is_neg_next;
        mul_done_r <= mul_done_next;
        state_r <= state_next;
        alu_mul_res_r <= alu_mul_res_next;
        op2_r <= op2_next;
        op1_r <= op1_next;
        acc_r <= acc_next;
    end
end

always @ ( * ) begin
    is_neg_next <= is_neg_r;
    mul_done_next <= mul_done_r;
    state_next <= state_r;
    alu_mul_res_next <= alu_mul_res_r;
    op2_next <= op2_r;
    op1_next <= op1_r;
    acc_next <= acc_r;
    case(state_r)
    IDLE:begin
        is_neg_next <= 1'b0;
        mul_done_next <= 32'b0;
        alu_mul_res_next <= 32'b0;
        op2_next <= 32'b0;
        op1_next <= 64'b0;
        acc_next <= 64'b0;
        if(alu_mul_stb_i)begin
        state_next <= INI;
        end
    end
    INI:begin
        op1_next <= alu_mul_op1_i[31]? {{32{1'b0}},-alu_mul_op1_i}:{{32{1'b0}},alu_mul_op1_i};
        op2_next <= alu_mul_op2_i[31]? -alu_mul_op2_i:alu_mul_op2_i;
        is_neg_next <= alu_mul_op1_i[31] ^ alu_mul_op2_i[31];
        state_next <= RUN;
    end
    RUN:begin
        if(op2_r == 32'b0)begin
        state_next <= DONE;
            if(alu_mul_funct_i == 5'b01110) begin
                acc_next <= is_neg_r? -acc_r:acc_r[31:0];
            end
            if(alu_mul_funct_i == 5'b01111) begin
                acc_next <= is_neg_r? -acc_r:acc_r;
            end
            if(alu_mul_funct_i == 5'b10000) begin
                acc_next <= alu_mul_op1_i[31]? -acc_r:acc_r;
            end
            if(alu_mul_funct_i == 5'b10001) begin
                acc_next <= acc_r; 
            end
        end
        else begin
            if(op2_r[0] == 1'b1)begin
            acc_next <= acc_r + op1_r;
            end
            op1_next <= op1_r << 1'b1;
            op2_next <= op2_r >> 1'b1;
            state_next <= RUN;
        end
    end
    DONE:begin
        if(alu_mul_funct_i == 5'b01110) begin
            alu_mul_res_next <= acc_r[31:0];
        end
        else begin
            alu_mul_res_next <= acc_r[63:32];
        end
        state_next <= IDLE;
        mul_done_next <= 1'b1;
    end
    endcase
end


endmodule
