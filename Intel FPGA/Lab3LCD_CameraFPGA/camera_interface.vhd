library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity camera_interface is
    port (
        clk     : in std_logic;
        n_reset : in std_logic;

        ----------------------
        -- Camera Interface --
        ----------------------
        -- Camera Interface <-> Avalon Master
        ci_RGB_pixel_out : out std_logic_vector(15 downto 0);
        ci_new_data      : out std_logic;
        ci_data_ack      : in std_logic;
        ci_new_frame     : out std_logic;

        -- Camera Interface <-> Avalon Slave
        ci_start : in std_logic;

        -- Camera Interface <-> Camera
        pixel_clk   : in std_logic;
        line_valid  : in std_logic;
        frame_valid : in std_logic;
        cam_data    : in std_logic_vector(11 downto 0);
        cam_reset_n : out std_logic;
        xclkin      : out std_logic
    );
end camera_interface;

architecture rtl of camera_interface is
    type acquisition_state is (cam_stop, cam_wait_frame, cam_frame, cam_line);
    type line_color is (GR, BG);
    type color is (green, red, blue);
    type transfer_state is (idle, send, trans_wait);

    signal acq_state       : acquisition_state;
    signal current_line    : line_color;
    signal current_color   : color;
    signal new_frame       : std_logic;
    signal prev_line_valid : std_logic;
    signal first_column    : std_logic;

    signal is_GR               : std_logic;
    signal is_BG               : std_logic;
    signal is_blue             : std_logic;
    signal is_not_first_column : std_logic;

    signal RGB_pixel : std_logic_vector(15 downto 0);

    signal merging_fifo_out   : std_logic_vector(11 downto 0);
    signal merging_fifo_empty : std_logic;
    signal merging_fifo_full  : std_logic;

    signal sync_fifo_empty : std_logic;
    signal sync_fifo_full  : std_logic;

    signal trans_state : transfer_state;

    signal G1 : std_logic_vector(11 downto 0);
    signal R  : std_logic_vector(11 downto 0);
    signal B  : std_logic_vector(11 downto 0);
    signal G2 : std_logic_vector(11 downto 0);

    component single_clk_fifo is
    port
    (
        clock : in std_logic;
        data  : in std_logic_vector (11 downto 0);
        rdreq : in std_logic;
        wrreq : in std_logic;
        empty : out std_logic;
        full  : out std_logic;
        q     : out std_logic_vector (11 downto 0)
    );
    END component single_clk_fifo;

    signal single_clk_read : std_logic;
    signal single_clk_write : std_logic;

    component dual_clk_fifo is
    port
    (
        data    : in std_logic_vector (15 downto 0);
        rdclk   : in std_logic;
        rdreq   : in std_logic;
        wrclk   : in std_logic;
        wrreq   : in std_logic;
        q       : out std_logic_vector (15 downto 0);
        rdempty : out std_logic;
        wrfull  : out std_logic 
    );
    end component dual_clk_fifo;

    signal dual_clk_read : std_logic;
    signal dual_clk_write : std_logic;
    
    component demosaicking is
    port
    (
        G1        : in std_logic_vector (11 downto 0);
        R         : in std_logic_vector (11 downto 0);
        B         : in std_logic_vector (11 downto 0);
        G2        : in std_logic_vector (11 downto 0);
        RGB_pixel : out std_logic_vector (15 downto 0)
    );
    end component demosaicking;

begin
    xclkin <= clk;
    ci_new_frame <= new_frame;

    is_GR <= '1' when (current_line = GR) else '0';
    is_BG <= '1' when (current_line = BG) else '0';
    is_blue <= '1' when (current_color = blue) else '0';
    is_not_first_column <= '1' when (first_column = '0') else '0';

    -- FIFO holding pixels from RG row waiting to be combined 
    pixel_merging_fifo : single_clk_fifo port map(
        clock => pixel_clk,
        data  => cam_data,
        rdreq => single_clk_read,
        wrreq => single_clk_write,
        empty => merging_fifo_empty,
        full  => merging_fifo_full,
        q     => merging_fifo_out
    );

    single_clk_read <= frame_valid and line_valid and is_BG;
    single_clk_write <= frame_valid and line_valid and is_GR;

    -- FIFO for synchronization purpose
    sync_fifo : dual_clk_fifo port map(
        data    => RGB_pixel,
        rdclk   => clk,
        rdreq   => dual_clk_read,
        wrclk   => pixel_clk,
        wrreq   => dual_clk_write,
        q       => ci_RGB_pixel_out,
        rdempty => sync_fifo_empty,
        wrfull  => sync_fifo_full
    );

    dual_clk_read <= '1' when (trans_state = send) else '0';
    dual_clk_write <= frame_valid and prev_line_valid and is_BG and is_blue and is_not_first_column;

    merger : demosaicking port map(
        G1        => G1,
        R         => R,
        B         => B,
        G2        => G2,
        RGB_pixel => RGB_pixel
    );

    -- FSM keeping track of current row and column
    frame_sweeping: process(pixel_clk, n_reset)
    begin
        if n_reset = '0' then
            acq_state <= cam_stop;
            cam_reset_n <= '0';
            new_frame <= '0';
        elsif rising_edge(pixel_clk) then
            case acq_state is
                when cam_stop =>
                    if ci_start = '1' then
                        acq_state <= cam_wait_frame;
                    end if;
                    cam_reset_n <= '1';
                when cam_wait_frame =>
                    if frame_valid = '1' then
                        acq_state <= cam_frame;
                        new_frame <= '1';
                    end if;
                when cam_frame =>
                    first_column <= '1';
                    acq_state <= cam_line;
                    new_frame <= '0';
                    current_line <= GR;
                    current_color <= green;
                when cam_line =>
                    prev_line_valid <= line_valid;
                    if frame_valid = '0' then
                        acq_state <= cam_wait_frame;
                    elsif line_valid = '1' then
                        if first_column = '1' then
                            first_column <= '0';
                        end if;
                        if current_line = GR then
                            if current_color = green then
                                current_color <= red;
                            elsif current_color = red then
                                current_color <= green;
                            end if;
                        else
                            if current_color = blue then
                                current_color <= green;
                                B <= cam_data;
                            elsif current_color = green then
                                current_color <= blue;
                                G2 <= cam_data;
                            end if;
                        end if;
                    -- transition from valid to invalid
                    elsif line_valid = '0' and prev_line_valid = '1' then
                        if current_line = GR then
                            current_line <= BG;
                            current_color <= blue;
                        else
                            current_line <= GR;
                            current_color <= green;
                        end if;
                    end if;
            end case;
        end if;
    end process;

    demuxing: process(merging_fifo_out, current_line, current_color)
    begin
        -- FIFO read would delay one cycle
        if current_line = BG and current_color = blue then
            R <= merging_fifo_out;
        elsif current_line = BG and current_color = green then
            G1 <= merging_fifo_out;
        end if;
    end process;

    -- FSM for transferring pixels to Avalon Master
    pixel_transfer: process(clk, n_reset)
    begin
        if n_reset = '0' then
            trans_state <= idle;
            ci_new_data <= '0';
        elsif rising_edge(clk) then
            case trans_state is
                when idle =>
                    if sync_fifo_empty = '0' then
                        trans_state <= send;
                    end if;
                when send =>
                    ci_new_data <= '1';
                    trans_state <= trans_wait;
                when trans_wait =>
                    ci_new_data <= '0';
                    -- wait until ack is received
                    if ci_data_ack = '1' then
                        trans_state <= idle;
                    end if;
            end case;
        end if;
    end process;
end;