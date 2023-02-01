library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multn is
    GENERIC (
        NBITS : positive
    );  
    port(
        opa,opb : in std_logic_vector (NBITS-1 downto 0); --operand A and B 
        clk : in std_logic; -- clk
        rst_b : in std_logic; -- reset signal active at 0
        stb : in std_logic; --indict when to start computation active at 1
        done : out std_logic; --indict when finish computation active at 1
        prod : out std_logic_vector (2*NBITS-1 downto 0) --production result
    );
    TYPE state_type IS (START,COMP,FINISH);
end entity multn;

architecture RTL of multn is
--------------------------------------------------------------------------------
--signals
--------------------------------------------------------------------------------
    constant all_zeros: std_logic_vector (NBITS-1 downto 0) := (others => '0');
    signal opa_r:  unsigned(2*NBITS-1 downto 0); --unsigned local copy of opa
    signal opb_r : unsigned(NBITS-1 downto 0); --unsigned local copy of opb
    signal acc : unsigned(2*NBITS-1 downto 0); --unsigned accumulator
    signal is_neg : std_logic; --negative result flag
    signal state: state_type;

begin
    process(clk,rst_b) --main process
    begin
        if rst_b = '0' then
            state <= START;
        else
            if rising_edge(clk) then
            case state is
                when START =>
                    if(stb = '1') then
                        state <= COMP;
                        acc <= (others => '0');
                        is_neg <= opa(31) xor opb(31);
                        opa_r <= unsigned(abs resize(signed(opa),NBITS*2));
                        opb_r <= unsigned(abs signed(opb));
                    done <= '0';
                    else
                        state <= START;
                    end if;
               when COMP =>
                    if(opb_r = to_unsigned(0,NBITS)) then
                        state <= FINISH;
                    elsif(opb_r(0) = '1') then
                        acc <= acc + unsigned(opa_r);
                        opa_r <= shift_left(opa_r,1);
                        opb_r <= shift_right(opb_r,1);
                        state <= COMP;
                    else
                        opa_r <= shift_left(opa_r,1);
                        opb_r <= shift_right(opb_r,1);
                        state <= COMP;
                    end if;
                when FINISH =>
                    state <= START;
            done <= '1';
                    if(is_neg = '1') then
                        prod <= std_logic_vector(-signed(acc));
                    else
                        prod <= std_logic_vector(acc);
                    end if;
            end case;
        end if;
        end if;
end process;
    


end architecture RTL;

