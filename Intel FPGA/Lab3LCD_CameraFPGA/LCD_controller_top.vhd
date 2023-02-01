library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

entity LCD_controller_top is
    port (
        clk         : in std_logic; -- main clock
        nReset  : in std_logic; -- active-low reset

        -- Avalon Slace (AS) interface
        AS_address          : in std_logic_vector(2 downto 0);
        AS_write            : in std_logic;
        AS_writedata        : in std_logic_vector(31 downto 0);
        AS_read                 : in std_logic;
        AS_readdata         : out std_logic_vector(31 downto 0);

        -- Avalon Master (AM) interface
        AM_address                  : out std_logic_vector(31 downto 0);
        AM_burstcount               : out std_logic_vector(6 downto 0);
        AM_read                         : out std_logic;
        AM_readdatavalid            : in std_logic;
        AM_readdata                 : in std_logic_vector(31 downto 0);
        AM_waitrequest          : in std_logic;

        -- Conduit to LT24 peripheral
        LCD_CS_N    : out std_logic;
        LCD_Reset_N : out std_logic;
        LCD_WR_N    : out std_logic;
        LCD_RD_N    : out std_logic;
        LCD_data    : out std_logic_vector(15 downto 0);
        LCD_on      : out std_logic;
        LCD_RS      : out std_logic
        );
end entity LCD_controller_top;

architecture comp of LCD_controller_top is
    -----------------------------------------------------------------------------
    -- FIFO component -----------------------------------------------------------
--    component fifo
--        port(
--        alcr		: IN STD_LOGIC ;
--        clock		: IN STD_LOGIC ;
--        data		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
--        rdreq		: IN STD_LOGIC ;
--        wrreq		: IN STD_LOGIC ;
--        almost_empty		: OUT STD_LOGIC ;
--        almost_full		: OUT STD_LOGIC ;
--        empty		: OUT STD_LOGIC ;
--        full		: OUT STD_LOGIC ;
--        q		: OUT STD_LOGIC_VECTOR (31 DOWNTO 0)
--    );
--    end component;

    -----------------------------------------------------------------------------
    -- Internal signals ---------------------------------------------------------
    -- AS_registers <=> LCD_controller interface
    signal Reg_LCD_en_i         : std_logic;
    signal Reg_LCD_cmd_data_i   : std_logic_vector(15 downto 0);
    signal Reg_LCD_cmd_or_data_en_i     : std_logic_vector(1 downto 0);
    signal LCD_Reg_ack_i        : std_logic;
    -- AS_registers <=> master_control
    signal AM_en_i          : std_logic;
    signal AM_rd_add_i  : std_logic_vector(31 downto 0);
    signal AM_rd_len_i  : std_logic_vector(31 downto 0);
    signal AM_burstcount_i : std_logic_vector(6 downto 0);
    -- master_control <=> FIFO_LCD
    signal FIFO_data_i  : std_logic_vector(31 downto 0);
    signal FIFO_wrreq_i : std_logic;
    signal FIFO_full_i  : std_logic;
    signal FIFO_al_full_i : std_logic;
    -- LCD_controller <=> FIFO_LCD
    signal FIFO_q_i         : std_logic_vector(31 downto 0);
    signal FIFO_empty_i : std_logic;
    signal FIFO_al_empty_i : std_logic;
    signal LCD_RdFIFO_i : std_logic;
    -- FIFO reset and used word
    signal Reset                : std_logic;


begin
    Reset <= not nReset;

    -----------------------------------------------------------------------------
    -- Instantiate and connect sub-modules
    REGISTERS : entity work.registers
    port map (
        clk         => clk,
        nReset  => nReset,
        Reg_LCD_cmd_data    => Reg_LCD_cmd_data_i,
        Reg_LCD_en  => Reg_LCD_en_i,
        Reg_LCD_cmd_or_data_en => Reg_LCD_cmd_or_data_en_i,
        LCD_Reg_ack         => LCD_Reg_ack_i,
        Reg_AM_en       => AM_en_i,
        Reg_AM_addr     => AM_rd_add_i,
        Reg_AM_lenth        => AM_rd_len_i,
        Reg_AM_burstcount   => AM_burstcount_i,
        address         => AS_address,
        write           => AS_write,
        writedata       => AS_writedata,
        read                => AS_read,
        readdata            => AS_readdata
		  );

    -----------------------------------------------------------------------------

    MASTER_CONTROL : entity work.master_control
    port map (
        clk         => clk,
        nReset  => nReset,
        reg_en      => AM_en_i,
        reg_lenth       => AM_rd_len_i,
        reg_addr        => AM_rd_add_i,
        reg_burstcount  => AM_burstcount_i,
        FIFO_data       => FIFO_data_i,
        FIFO_wrreq      => FIFO_wrreq_i,
        FIFO_full       => FIFO_full_i,
        FIFO_al_full    => FIFO_al_full_i,
        address                     => AM_address,
        burstcount              => AM_burstcount,
        read                        => AM_read,
        readdatavalid           => AM_readdatavalid,
        readdata                => AM_readdata,
        waitrequest             => AM_waitrequest
        );

    -----------------------------------------------------------------------------

    LCD_CONTROL: entity work.LCD_control
    port map (
        clk              => clk,
        nReset       => nReset,
        Reg_LCD_en  => Reg_LCD_en_i,
        Reg_LCD_cmd_or_data_en   => Reg_LCD_cmd_or_data_en_i,
        Reg_LCD_cmd_data => Reg_LCD_cmd_data_i,
        LCD_Reg_ack          => LCD_Reg_ack_i,
        FIFO_data    => FIFO_q_i,
        FIFO_empty   => FIFO_empty_i,
        FIFO_Almost_empty => FIFO_al_empty_i,
        LCD_RdFIFO   => LCD_RdFIFO_i,
        LCD_CS_n     => LCD_CS_N,
        LCD_Reset_n          => LCD_Reset_N,
        LCD_WR_n         => LCD_WR_N,
        LCD_RD_n     => LCD_RD_N,
        LCD_D    => LCD_data,
        LCD_on  => LCD_on,
        LCD_RS => LCD_RS
        );

    -----------------------------------------------------------------------------

    FIFO : entity work.fifo PORT MAP (
        clock    => clk,
        data    => FIFO_data_i,
		  rdreq  => LCD_RdFIFO_i,
        wrreq  => FIFO_wrreq_i,
        almost_empty    => FIFO_al_empty_i,
        almost_full     => FIFO_al_full_i,
        empty    => FIFO_empty_i,
        full     => FIFO_full_i,
		  q   => FIFO_q_i
    );

end architecture comp;