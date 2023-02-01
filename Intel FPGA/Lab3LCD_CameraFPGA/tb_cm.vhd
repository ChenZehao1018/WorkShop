library ieee; 
use ieee.std_logic_1164.all; 
use ieee.numeric_std.all; 
 
entity tb_am is 
end tb_am; 
 
architecture test of tb_am is 
    signal clk: std_logic := '0'; 
    signal n_reset: std_logic := '1'; 
 
    signal as_address : std_logic := '0';
    signal as_write : std_logic := '0';
    signal as_write_data : std_logic_vector(31 downto 0) := (others => '0');

    signal as_start : std_logic;

    signal am_address      :  std_logic_vector(31 downto 0);
    signal am_byte_enable  :  std_logic_vector(3 downto 0);
    signal am_burst_count  :  std_logic_vector(3 downto 0); -- modify graph
    signal am_write        :  std_logic;
    signal am_write_data   :  std_logic_vector(31 downto 0);
    signal am_wait_request :  std_logic;

    signal am_pixel_transfer : std_logic_vector(15 downto 0);
    signal am_new_data       : std_logic;
    signal am_data_ack       : std_logic;
    signal am_new_frame      : std_logic;


    constant clk_period : time := 20 ns;

begin 
    dut: entity work.camera_avalon_manager
        port map( 
            clk => clk, 
            n_reset => n_reset,

            as_address => as_address,
            as_write => as_write,
            as_write_data => as_write_data,

            as_start => as_start,

            am_address => am_address,
            am_byte_enable => am_byte_enable,
            am_burst_count => am_burst_count,
            am_write => am_write,
            am_write_data => am_write_data,
            am_wait_request => am_wait_request,

            am_pixel_transfer => am_pixel_transfer,
            am_new_data => am_new_data,
            am_data_ack => am_data_ack,
            am_new_frame => am_new_frame
        ); 
 
    clk <= not clk after (clk_period / 2);

    simulation: process 
    begin 
        -- reset everything 
        n_reset <= '0';
        wait for clk_period;
        n_reset <= '1';

        -- set start_address
        as_address <= '0';
        as_write <= '1';
        as_write_data <= "00000000000000000000000000000000";
        wait until rising_edge(clk);

        -- set start
        as_address <= '1';
        as_write <= '1';
        as_write_data <= "00000000000000000000000000000001";
        wait until rising_edge(clk);
        
        -- new frame
        am_pixel_transfer <= "1000100010001000";

        for i in 0 to 3 loop
            for pixel_id in 0 to 10 loop 
                wait until rising_edge(clk);
                if pixel_id = 0 then
                    am_new_frame <= '1';
                    wait until rising_edge(clk);
                    am_new_frame <= '0';
                else
                    am_new_frame <= '0';
                end if;

                am_new_data <= '1';
                am_pixel_transfer <= std_logic_vector(to_unsigned(pixel_id, 16));
                
                am_wait_request <= '1';
                wait until rising_edge(clk);
                wait until rising_edge(clk);

                am_wait_request <= '0';
                wait until am_data_ack = '1';

                am_new_data <= '0';
                wait until rising_edge(clk);

            end loop;
        end loop;
        
 
        wait; 
    end process simulation; 
 
end architecture test;
