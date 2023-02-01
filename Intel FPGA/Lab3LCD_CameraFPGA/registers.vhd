library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity registers is Port (
    --global signal
    clk : in std_logic;
    nReset : in std_logic;
     
    --Internalin terface(i.e.Avalon slave).
    address : in std_logic_vector(2 downto 0);
    write : in std_logic;
    read : in std_logic;
    writedata : in std_logic_vector(31 downto 0);
    readdata : out std_logic_vector(31 downto 0);

    --LCD controller interface(i.e.LCD).
     Reg_LCD_en : out std_logic;
     Reg_LCD_cmd_or_data_en : out std_logic_vector(1 downto 0);
     LCD_Reg_ack : in std_logic;
     Reg_LCD_cmd_data : out std_logic_vector(15 downto 0);
     
     --master controller interface
     Reg_AM_en : out std_logic;
     Reg_AM_addr : out std_logic_vector(31 downto 0);
     Reg_AM_burstcount : out std_logic_vector(6 downto 0);
     Reg_AM_lenth : out std_logic_vector(31 downto 0)
     );
end registers;

architecture Behavioral of registers is
    signal Reg_AM_addr_s : std_logic_vector(31 downto 0);
    signal Reg_AM_len_s : std_logic_vector(31 downto 0);
    signal Reg_AM_burstcount_s : std_logic_vector(6 downto 0);
    signal Reg_AM_en_s : std_logic;
    signal Reg_LCD_en_s : std_logic;
    signal Reg_LCD_cmd_data_s : std_logic_vector(15 downto 0);
    signal Reg_LCD_cmd_or_data_en_s : std_logic_vector(1 downto 0);
begin

    --send signal
    send_signal:process(clk,nReset) is
    begin
        if nReset ='0' then
            Reg_AM_addr     <= (others =>'0');
            Reg_AM_lenth    <= (others =>'0');
            Reg_AM_burstcount   <= (others =>'0');
            Reg_AM_en       <= '0';
            Reg_LCD_en      <= '0';
            Reg_LCD_cmd_data    <= (others =>'0');
            Reg_LCD_cmd_or_data_en  <= (others =>'0');
        elsif rising_edge(clk) then 
            Reg_AM_addr         <=      Reg_AM_addr_s;
            Reg_AM_lenth        <=      Reg_AM_len_s;
            Reg_AM_burstcount   <=      Reg_AM_burstcount_s;
            Reg_AM_en           <=      Reg_AM_en_s;
            Reg_LCD_en          <=      Reg_LCD_en_s;
            if LCD_Reg_ack  =   '1' then
                Reg_LCD_cmd_data    <=      Reg_LCD_cmd_data_s;
                Reg_LCD_cmd_or_data_en  <=  Reg_LCD_cmd_or_data_en_s;
            end if;
        end if;
    end process;

    --slave read
    slave_read: process (clk) is
    begin
        if rising_edge(clk) then
            if read='1' then 
                case address is
                        when "000" => readdata(31 downto 0)     <= Reg_AM_addr_s;
                        when "001" => readdata(31 downto 0)     <= Reg_AM_len_s;
                        when "010" => readdata(6 downto 0)      <= Reg_AM_burstcount_s;
                        when "011" => readdata(0)               <= Reg_LCD_en_s;
                        when "100" => readdata(15 downto 0)     <= Reg_LCD_cmd_data_s;
                        when "101" => readdata(1 downto 0)      <= Reg_LCD_cmd_or_data_en_s;
                        when "110" => readdata(0)               <= Reg_AM_en_s;
                        when others => null;--writedata(0) <= Reg_en;
                end case;
            end if;            
        end if;
    end process;
    
    --slave write
    slave_write: process(clk,nReset) is
    begin
        if nReset='0' then
            Reg_AM_addr_s       <= (others =>'0');
            Reg_AM_len_s        <= (others =>'0');
            Reg_AM_burstcount_s <= (others =>'0');
            Reg_AM_en_s         <= '0';
            Reg_LCD_en_s        <= '0';
            Reg_LCD_cmd_data_s  <= (others =>'0');
            Reg_LCD_cmd_or_data_en_s    <= (others =>'0');
        elsif rising_edge(clk) then
            if write='1' then
                case Address is
                        when "000" => Reg_AM_addr_s         <= writedata;
                        when "001" => Reg_AM_len_s          <= writedata;
                        when "010" => Reg_AM_burstcount_s   <= writedata(6 downto 0);
                        when "110" => Reg_AM_en_s           <= writedata(0);
                        when "011" => Reg_LCD_en_s          <= writedata(0);
                        when "100" => Reg_LCD_cmd_data_s    <= writedata(15 downto 0);
                        when "101" => Reg_LCD_cmd_or_data_en_s  <= writedata(1 downto 0);
                        when others => null;--Reg_en <= writedata(0);
                 end case;
            end if;
        end if;
    end process;       
end Behavioral;