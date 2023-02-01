library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_ci is
end tb_ci;

architecture test of tb_ci is
    signal clk: std_logic := '0';
    signal n_reset: std_logic := '1';

    signal ci_RGB_pixel_out : std_logic_vector(15 downto 0);
    signal ci_new_data : std_logic;
    signal ci_data_ack : std_logic;
    signal ci_new_frame : std_logic;
    signal ci_start : std_logic;
    signal pixel_clk : std_logic;
    signal line_valid : std_logic;
    signal frame_valid : std_logic;
    signal cam_data : std_logic_vector(11 downto 0);
    signal cam_reset_n : std_logic;
    signal xclkin : std_logic;

    constant clk_period : time := 20 ns;

    constant row : integer := 2;
    constant col : integer := 3;

    signal R : std_logic_vector(11 downto 0);
    signal B : std_logic_vector(11 downto 0);
    signal G_sum : std_logic_vector(11 downto 0);
begin
    dut: entity work.camera_interface
        port map(
            clk => clk,
            n_reset => n_reset,
            ci_RGB_pixel_out => ci_RGB_pixel_out,
            ci_new_data => ci_new_data,
            ci_data_ack => ci_data_ack,
            ci_new_frame => ci_new_frame,
            ci_start => ci_start,
            pixel_clk => pixel_clk,
            line_valid => line_valid,
            frame_valid => frame_valid,
            cam_data => cam_data,
            cam_reset_n => cam_reset_n,
            xclkin => xclkin
        );

    clk <= not clk after (clk_period / 2);
    pixel_clk <= clk;

    simulation: process

        -- procedure for generating pixel signals
        procedure generate_pixel(val: in std_logic_vector(11 downto 0)) is
        begin
            cam_data <= val;
            wait until rising_edge(pixel_clk);
        end procedure;

    begin
        -- reset everything
        n_reset <= '0';
        ci_start <= '0';
        line_valid <= '0';
        frame_valid <= '0';
        ci_data_ack <= '0';
        cam_data <= (others => '1');
        wait for clk_period;
        n_reset <= '1';
        ci_start <= '1';
        ci_data_ack <= '1';

        wait until rising_edge(pixel_clk);
        frame_valid <= '1';
        for i in 0 to row - 1 loop
            -- GR row
            wait until rising_edge(pixel_clk);
            line_valid <= '1';
            for j in 0 to col - 1 loop
                -- G1
                generate_pixel(std_logic_vector(to_unsigned(i * col + j, 12)));
                -- R
                generate_pixel(std_logic_vector(to_unsigned(i * col + j + 1, 12)));
            end loop;
            line_valid <= '0';
            wait until rising_edge(pixel_clk);

            -- BG row
            wait until rising_edge(pixel_clk);
            line_valid <= '1';
            for j in 0 to col - 1 loop
                -- B
                generate_pixel(std_logic_vector(to_unsigned(i * col + j + 2, 12)));
                -- G2
                generate_pixel(std_logic_vector(to_unsigned(i * col + j + 3, 12)));
                R <= std_logic_vector(to_unsigned(i * col + j + 1, 12));
                B <= std_logic_vector(to_unsigned(i * col + j + 2, 12));
                G_sum <= std_logic_vector(to_unsigned(i * col + j, 12) + to_unsigned(i * col + j + 3, 12));
            end loop;
            line_valid <= '0';
            wait until rising_edge(pixel_clk);
        end loop;
        frame_valid <= '0';
        wait until rising_edge(pixel_clk);

        wait;
    end process simulation;

end architecture test;

