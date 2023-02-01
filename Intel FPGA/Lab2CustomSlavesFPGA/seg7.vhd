library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity seg7 is port(
    clk : in std_logic;
    nReset : in std_logic;
    address : in std_logic_vector(2 downto 0);
    write : in std_logic;
    read : in std_logic;
    writedata : in std_logic_vector(7 downto 0);
    
    readdata : out std_logic_vector(7 downto 0);
    selSeg : out std_logic_vector(7 downto 0);
    nSelDig: out std_logic_vector(5 downto 0);
    Reset_Led: out std_logic
);
end seg7;

architecture behavior of seg7 is
    type num_type is array (5 downto 0) of std_logic_vector(4 downto 0);
    type carry_type is array (0 to 5) of integer range 0 to 9;
    type seg7_type is array (0 to 9) of std_logic_vector(6 downto 0);
    type selSeg_array_type is array (5 downto 0) of std_logic_vector(7 downto 0);
 
    constant dot_mask : std_logic_vector(5 downto 0) := "010100"; --used to mark the dot
    constant carry : carry_type := (9, 9, 9, 5, 9, 9); --used to give carry of rtc
    constant seg7 : seg7_type :=("1111110", "0110000", "1101101", "1111001", "0110011", "1011011", "1011111", "1110000", "1111111", "1111011");
    
    signal control_reg : std_logic_vector(5 downto 0);
    signal reset_led_reg : std_logic;
    signal clk_6khz : std_logic;  --clk of 6khz T 1/6ms
    signal sel_trans_flag : std_logic;
    signal selDig_reg: integer range 0 to 5;
    signal nSelDig_reg: std_logic_vector(5 downto 0);
    signal store_reg : num_type;
    signal rtc_array_reg : num_type;
    signal selSeg_array: selSeg_array_type;
    signal reset_led_counter : integer range 0 to 5;
    signal rtc_flag : std_logic;
begin

p1_slowClock:process(nReset, clk)
    variable i : integer range 0 to 8332;
begin
    if nReset = '0' then
        i := 0;
    elsif rising_edge(clk) then
        i := (i + 1) mod 8333;
        if i = 8331 then
            clk_6khz <= '1';
        else
            clk_6khz <= '0';
        end if;
    end if;
end process p1_slowClock;

p2_led_reset:process(nReset, clk)
begin
    if nReset = '0' then
        reset_led_counter <= 0;
    elsif rising_edge(clk) then
        if clk_6khz = '1' then
            reset_led_counter <= (reset_led_counter + 1) mod 6;
        end if;
    end if;
end process p2_led_reset;

p3_seg7_sel:process(nReset, clk)
    variable i : integer range 0 to 5 := 0;
begin
    if nReset = '0' then
        i := 0;
    elsif rising_edge(clk) then
        if sel_trans_flag = '1' then
            i := (i + 1) mod 6;
        end if;
    end if;
    selDig_reg <= i;
end process p3_seg_sel;

p4_seg7_prog:process(nReset, clk)
    variable i : integer range 0 to 499999;
    variable carry_flag: boolean;
begin
    if nReset = '0' then
        i := 0;
rtc_array_reg <= (others => (others => '0'));
    elsif rising_edge(clk) then
        i := (i + 1) mod 500000;
        if i = 499999 then
            carry_flag := true;
            for j in 0 to 5 loop
                if carry_flag = true then
                    if to_integer(unsigned(rtc_array_reg(j))) = carry(j) then
                        rtc_array_reg(j) <= (others => '0');
                    else
                        rtc_array_reg(j) <= std_logic_vector(unsigned(rtc_array_reg(j)) + 1);
                        carry_flag := false;
                    end if;
                end if;
            end loop;
        end if;
    end if;
end process p4_seg7_prog;


p5_write:process(clk,nReset)
begin
    if nReset = '0' then
        store_reg <= (others => (others => '0'));
        mask_reg  <= (others => '0');
        rtc_flag <= '1';
    elsif rising_edge(clk) then
        if write = '1' then
            case address is
                when "000" => control_reg <= writedata(5 downto 0);
                when others=>
                    if (to_integer(unsigned(address)) <= 6+1) then
                       store_reg(to_integer(unsigned(address))-1) <= writedata(4 downto 0);
                    elsif (to_integer(unsigned(address)) = 8) then
                        rtc_flag <= writedata(0);
                    end if;
            end case;
        end if;
    end if;
end process p5_write;

p6_read:process(clk)
begin
    if rising_edge(clk) then
        readdata <= (others=>'0');
        if read = '1' then
            case address is
                when "000" => readdata(5 downto 0) <= control_reg;
                when others=>
                    if (to_integer(unsigned(address)) <= 6+1) then
                       readdata(4 downto 0) <= store_reg(to_integer(unsigned(address))-1);
                    elsif (to_integer(unsigned(address)) = 8) then
                        readdata(0) <= rtc_flag;
                    end if;
            end case;
        end if;
    end if;
end process p6_read;

num_select: for i in 0 to 5 generate
    nSelDig_reg(i) <= '0' when (selDig_reg = i) else '1';
end generate;
nSelDig <= nSelDig_reg when reset_led_reg = '0' else (others => '1');

seg7_lookup: for i in 0 to 5 generate
    selSeg_array(i) <= seg7(to_integer(unsigned(store_reg(i)))) & control_reg(i) when rtc_flag='0' else
                       seg7(to_integer(unsigned(rtc_array_reg(i)))) & dot_mask(i);
end generate;

seg7_assign: for i in 0 to 7 generate
    selSeg(i) <= selSeg_array(selDig_reg)(7-i);
end generate;

reset_led_reg <= '1' when reset_led_counter = 0 else '0';
Reset_Led <= reset_led_reg;
sel_trans_flag <= '1' when reset_led_counter = 0 and clk_6khz='1' else '0';

end behavior;