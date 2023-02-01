library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use ieee.numeric_std.all;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity master_control is
    port(
        --Global signals
        clk : in std_logic;
        nReset : in std_logic;
       
        --Register
        reg_addr : in std_logic_vector(31 downto 0);
        reg_burstcount : in std_logic_vector(6 downto 0);
        reg_lenth : in std_logic_vector(31 downto 0);
        reg_en : in std_logic;
       
        --FIFO
        FIFO_data : out std_logic_vector(31 downto 0);
        FIFO_wrreq : out std_logic;
        FIFO_full : in std_logic;
        FIFO_al_full : in std_logic;

        --Avalon Master (AM) interface
        address : out std_logic_vector(31 downto 0);
        burstcount : out std_logic_vector(6 downto 0);
        read : out std_logic;
        readdatavalid : in std_logic;
        readdata : in std_logic_vector(31 downto 0);
        waitrequest : in std_logic
);
end master_control;

architecture Behavioral of master_control is

type state_type is (IDLE, BurstStart, BurstWait, BurstRead);

--Internal Registers
signal state_reg, state_next : state_type;
signal FIFO_data_reg, FIFO_data_next : std_logic_vector(31 downto 0);
signal FIFO_wrreq_reg, FIFO_wrreq_next : std_logic;
signal address_reg, address_next : std_logic_vector(31 downto 0);
signal start_addr_reg, start_addr_next : unsigned(31 downto 0);
signal burstcount_reg, burstcount_next : unsigned(8 downto 0);
signal burstcount_out_reg, burstcount_out_next : std_logic_vector(6 downto 0);
signal read_reg, read_next : std_logic;
signal word_cnt_reg, word_cnt_next : unsigned(16 downto 0);
signal burst_cnt_reg, burst_cnt_next : unsigned(8 downto 0);
signal length_reg, length_next : unsigned(31 downto 0);

begin
        FIFO_data <= FIFO_data_reg;
        FIFO_wrreq <= FIFO_wrreq_reg;
        address <= address_reg;
        burstcount <= burstcount_out_reg;
        read <= read_reg;


    process(clk,nReset)
    begin
        if nReset = '0' then
            state_reg <= IDLE;
            FIFO_data_reg <= (others => '0');
            FIFO_wrreq_reg <= '0';
            address_reg <= (others => '0');
            start_addr_reg <= (others => '0');
            burstcount_reg <= (others => '0');
            burstcount_out_reg <= (others => '0');
            read_reg <= '0';
            word_cnt_reg <= (others => '0');
            burst_cnt_reg <= (others => '0');
            length_reg <= (others => '0');
           
        elsif rising_edge(clk) then
            state_reg <= state_next;
            FIFO_data_reg <= FIFO_data_next;
            FIFO_wrreq_reg <= FIFO_wrreq_next;
            address_reg <= address_next;
            start_addr_reg <= start_addr_next;
            burstcount_reg <= burstcount_next;
            burstcount_out_reg <= burstcount_out_next;
            read_reg <= read_next;
            word_cnt_reg <= word_cnt_next;
            burst_cnt_reg <= burst_cnt_next;
            length_reg <= length_next;
        end if;
    end process;
    process(state_reg,reg_en,waitrequest,FIFO_al_full,burst_cnt_reg, word_cnt_reg, readdatavalid)
    begin
        state_next <= state_reg;
        FIFO_data_next <= FIFO_data_reg;
        FIFO_wrreq_next <= FIFO_wrreq_reg;
        address_next <= address_reg;
        start_addr_next <= start_addr_reg;
        burstcount_next <= burstcount_reg;
        burstcount_out_next <= burstcount_out_reg;
        read_next <= read_reg;
        word_cnt_next <= word_cnt_reg;
        burst_cnt_next <= burst_cnt_reg;
        length_next <= length_reg;

        case state_reg is
            when IDLE =>
                FIFO_data_next <= (others => '0');
                FIFO_wrreq_next <= '0';
                address_next <= (others => '0');
                start_addr_next <= (others => '0');
                burstcount_next <= (others => '0');
                burstcount_out_next <= (others => '0');
                read_next <= '0';
                word_cnt_next <= (others => '0');
                burst_cnt_next <= (others => '0');
                length_next <= (others => '0');

                if reg_en = '1' then
                    start_addr_next <= unsigned(reg_addr);
                    burstcount_next(6 downto 0) <= unsigned(reg_burstcount);
                    length_next <= unsigned(reg_lenth);
                    state_next <= BurstStart;

                end if;

            when BurstStart =>
                FIFO_wrreq_next <= '0';

                if FIFO_al_full = '1' or FIFO_full = '1' then
                    state_next <= BurstStart;
                else
                    read_next <= '1';
                    address_next <= std_logic_vector(start_addr_reg);
                    burstcount_out_next <= std_logic_vector(burstcount_reg(6 downto 0));
                    state_next <= BurstWait;
                end if;

            when BurstWait =>

                if waitrequest = '1' then
                    state_next <= BurstWait;
                else
                    read_next <= '0';
                    address_next <= std_logic_vector(start_addr_reg);
                    burstcount_out_next <= (others => '0');
                    state_next <= BurstRead;
                end if;

            when BurstRead =>
                if readdatavalid = '1' then
                    FIFO_wrreq_next <= '1';
                    FIFO_data_next <= readdata;
                    burst_cnt_next <= burst_cnt_reg + 1;
                    word_cnt_next <= word_cnt_reg + 1;

                    if word_cnt_reg = length_reg-1 or burstcount_reg = 0 then -- whole transfer finished
                        state_next <= IDLE;

                    elsif burst_cnt_reg = burstcount_reg(6 downto 0)-1 then -- one burst done
                        start_addr_next <= start_addr_reg + shift_left(burstcount_reg, 2);
                        burst_cnt_next <= (others => '0');
                        word_cnt_next <= word_cnt_reg+1;
                        state_next <= BurstStart;
                    end if;

                else
                    FIFO_wrreq_next <= '0';
                    state_next <= BurstRead;
                end if;

            when others => null;

        end case;
    end process;


end Behavioral;