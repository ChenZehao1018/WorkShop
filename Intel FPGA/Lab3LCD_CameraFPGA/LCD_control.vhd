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

entity LCD_control is Port ( 
    --global signal
    clk: in std_logic;
    nReset: in std_logic;
    --registers interface
    Reg_LCD_en: in std_logic;--turn on or off thr LCD
    Reg_LCD_cmd_data: in std_logic_vector(15 downto 0);--data or cmd to write
    Reg_LCD_cmd_or_data_en: in std_logic_vector(1 downto 0);--01 write cmd; 10 write data; 00 no operatiom
    LCD_Reg_ack: out std_logic;
    --FIFO interface
    FIFO_data: in std_logic_vector(31 downto 0);
    FIFO_empty: in std_logic;
    FIFO_Almost_empty: in std_logic;
    LCD_RdFIFO: out std_logic;
    --LCD interface
    LCD_on: out std_logic;
    LCD_Reset_n: out std_logic;
    LCD_CS_n: out std_logic;
    LCD_RS: out std_logic;--DC:L:command H:data
    LCD_RD_n: out std_logic;
    LCD_WR_n: out std_logic;
    LCD_D: out std_logic_vector(15 downto 0));
end LCD_control;

architecture Behavioral of LCD_control is
    type state_type is (IDLE,send_cmd,fifo_wait,send_pix,send_cmd_data);
    signal LCD_state: state_type;
    
    constant PIX_PER_LINE : unsigned(7 downto 0) := to_unsigned(240,8);
    constant CMD_MEM_WRITE : std_logic_vector(15 downto 0) := std_logic_vector(to_unsigned(16#002C#, 16));
    constant IMG_WIDTH      : positive := 320;
    constant IMG_HEIGHT     : positive := 240;
    constant IMG_WORD_LEN   : positive := IMG_WIDTH*IMG_HEIGHT;
    signal cnt              : natural range 0 to IMG_WORD_LEN;
    signal cnt_cycle        : natural range 0 to 7;  
    signal FIFO_data_orginal:  std_logic_vector(31 downto 0);
    signal Reg_LCD_cmd_or_data_en_s: std_logic_vector(1 downto 0);
begin
    --state change(FSM)
    state: process (clk,nReset) is
    begin
        if nReset ='0' then
            LCD_state   <=  idle;
            cnt         <=  0;
            cnt_cycle   <=  0;
            LCD_Reg_ack <=  '0';

            LCD_Reset_n <=  '0';
            LCD_CS_n    <=  '1';
            LCD_RS      <=  '1';
            LCD_RD_n    <=  '1';
            LCD_WR_n    <=  '1';
            LCD_D       <= (others => 'Z');
        elsif rising_edge(clk) then
            Reg_LCD_cmd_or_data_en_s <= Reg_LCD_cmd_or_data_en;
            LCD_on  <=  Reg_LCD_en;
            case LCD_state is
                when idle       =>
                    cnt         <= 0;
                    cnt_cycle   <=  0;
                    
                    LCD_Reset_n <=  '1';
                    LCD_CS_n    <=  '1';
                    LCD_RS      <=  '1';
                    LCD_RD_n    <=  '1';
                    LCD_WR_n    <=  '1';
                    --LCD_D       <= (others => 'Z');
                    
                    LCD_RdFIFO  <= '0';
                    LCD_Reg_ack <=  '1';
                    if Reg_LCD_en='1' and Reg_LCD_cmd_or_data_en_s="01" then
                        LCD_state   <= send_cmd;
                        LCD_Reg_ack <=  '0';
                    elsif Reg_LCD_en='1' and Reg_LCD_cmd_or_data_en_s="10" then
                        LCD_state   <= send_cmd_data;
                        LCD_Reg_ack <=  '0'; 
                    else   LCD_state   <= idle;                 
                    end if;
                ------------------------------------------------------------------------------------------------
                when send_cmd   =>
                    cnt_cycle   <=  cnt_cycle + 1;
                    case cnt_cycle is
                        when 0  =>
                            LCD_CS_n    <=  '0';
                            LCD_RS      <=  '0';
                            LCD_WR_n    <=  '0';
                            LCD_D       <=  Reg_LCD_cmd_data;
                        when 1  =>
                        when 2  =>  LCD_WR_n    <=  '1';
                        when others  =>
                            LCD_CS_n    <=  '1';
                            LCD_RS      <=  '1';
                            cnt_cycle   <=  0;
                            if Reg_LCD_cmd_data = CMD_MEM_WRITE then
                                LCD_state   <=  fifo_wait;
                                LCD_Reg_ack <=  '1';
                            else
                                LCD_state   <=  idle;
                                LCD_Reg_ack <=  '1';
                            end if;
                    end case;
                ---------------------------------------------------------------------------------------------------
                when fifo_wait  =>
                    --if Reg_LCD_cmd_or_data_en="01" then
                        --LCD_state   <=  send_cmd;
                        --LCD_Reg_ack <=  '0';
                    --els
                    if   FIFO_almost_empty ='0' and FIFO_empty= '0'then
                        LCD_Reg_ack <=  '0';
                        LCD_RdFIFO  <= '1';
                        FIFO_data_orginal   <= FIFO_data;
                        LCD_state   <=  send_pix;
                    else    LCD_state   <=  fifo_wait;
                    end if;
                ------------------------------------------------------------------
                when send_pix   =>
                    cnt_cycle   <=  cnt_cycle + 1;
                    case cnt_cycle is
                        when 0|4    =>
                            LCD_CS_n    <=  '0';
                            LCD_RS      <=  '1';
                            LCD_WR_n    <=  '0';
                            if cnt_cycle=0 then 
                                LCD_D   <=  FIFO_data_orginal(15 downto 0);
                            else
                                LCD_D   <=  FIFO_data_orginal(31 downto 16);
                            end if;
                            LCD_RdFIFO  <='0';
                        when 1|5    =>  null;
                        when 2|6    =>  LCD_WR_n    <=  '1';
                        when 3      =>
                            LCD_CS_n    <=  '1';
                            cnt <=  cnt+1;
                        when others =>
                            LCD_CS_n    <=  '1';
                            cnt_cycle   <=  0;
                            if cnt = PIX_PER_LINE-1 then
                                cnt <=  0;
                                LCD_CS_n    <=  '1';
                                LCD_RS      <=  '1';
                                LCD_state   <=  fifo_wait;
                                LCD_Reg_ack <=  '1';
                            elsif FIFO_almost_empty ='1' or FIFO_empty= '1' then
                                LCD_state   <=  fifo_wait;
                            elsif Reg_LCD_en='0'then
                                LCD_state   <=  idle;
                            else 
                                cnt <=  cnt + 1;
                                LCD_RdFIFO  <= '1';
                                FIFO_data_orginal   <= FIFO_data;
                                LCD_state   <=  send_pix;
                            end if;
                    end case;
                -----------------------------------------------------------------------------------------------------
                when send_cmd_data=>
                    cnt_cycle   <=  cnt_cycle + 1;
                    case cnt_cycle is
                        when 0  =>
                            LCD_CS_n    <=  '0';
                            LCD_RS      <=  '1';
                            LCD_WR_n    <=  '0';
                            LCD_D       <=  Reg_LCD_cmd_data;
                        when 1  =>
                        when 2  =>  LCD_WR_n    <=  '1';
                        when others  =>
                            LCD_CS_n    <=  '1';
                            LCD_RS      <=  '1';
                            cnt_cycle   <=  0;
                            LCD_state   <=  idle;
                            LCD_Reg_ack <=  '1';
                    end case;
                when others     => null;
            end case;            
        end if;
    end process;


end Behavioral;