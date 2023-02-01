library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity camera_avalon_manager is
    port (
        clk     : in std_logic;
        n_reset : in std_logic;

        ------------------
        -- Avalon Slave --
        ------------------
        -- Avalon Slave <-> Avalon Bus
        as_address    : in std_logic; -- 0 : start_address, 1 : start
        as_write      : in std_logic;
        as_write_data : in std_logic_vector(31 downto 0);

        -- Avalon Slave <-> Avalon Master
        -- as_start_address : out std_logic_vector(31 downto 0);

        -- Avalon Slave <-> Camera Interface
        as_start : out std_logic;

        -------------------
        -- Avalon Master --
        -------------------
        -- Avalon Master <-> Avalon Bus
        am_address      :  out std_logic_vector(31 downto 0);
        am_byte_enable  :  out std_logic_vector(3 downto 0);
        am_burst_count  :  out std_logic_vector(3 downto 0); -- modify graph
        am_write        :  out std_logic;
        am_write_data   :  out std_logic_vector(31 downto 0);
        am_wait_request :  in std_logic;

        -- Avalon Master <-> Camera Interface
        am_pixel_transfer : in std_logic_vector(15 downto 0);
        am_new_data       : in std_logic;
        am_data_ack       : out std_logic;
        am_new_frame      : in std_logic

        -- Avalon Master <-> Avalon Slave
        -- am_start         : in std_logic
    );
end camera_avalon_manager;

architecture RTL of camera_avalon_manager is
    signal frame_buffer_flag : boolean := true; -- using 0 / 1 buffer, init as 1, so the first frame will make it 0
    signal current_offset : unsigned(31 downto 0);
    signal as_start_address : std_logic_vector(31 downto 0);
    signal wait_counter : unsigned(3 downto 0);
begin

am_write_data(31 downto 16) <= (others => '0');
am_write_data(15 downto 0) <= am_pixel_transfer;

am_byte_enable <= "0011";
am_burst_count <= "0001";

-------------------
-- Avalon Master --
-------------------
-- Read Pixel: read pixel from camera interface and write to avalon bus
read_pixel : process(clk, n_reset)
begin
    if n_reset = '0' then
        am_data_ack <= '0';

        wait_counter <= (others => '0');

        frame_buffer_flag <= false;
        current_offset <= (others => '0');

    elsif rising_edge(clk) then
        if am_new_frame = '1' then
            frame_buffer_flag <= not frame_buffer_flag;
            if frame_buffer_flag = true then
                current_offset <= to_unsigned(153600, 32);
            else
                current_offset <= (others => '0');
            end if;
        end if;

        if am_new_data = '1' and am_wait_request = '0' then
            am_write <= '1';
            am_address <= std_logic_vector(unsigned(as_start_address) + unsigned(current_offset));
            wait_counter <= wait_counter + 1;

            if wait_counter = "0001" then
                wait_counter <= (others => '0');
                am_write <= '0';
                am_data_ack <= '1';
                current_offset <= current_offset + 2;
            end if;
        
        else 
            am_write <= '0';
            am_data_ack <= '0';
        
        end if;
    end if;
end process read_pixel;

-- Update Start address: when am_start is high, update current address
-- update_start_address : process(clk, n_reset)
-- begin
--     if n_reset = '0' then
--         base_address <= (others => '0');
--         current_offset <= (others => '0');
--     elsif rising_edge(clk) then
--         if am_start = '1' then
--             base_address <= am_start_address;
--         end if;
--     end if;
-- end process update_start_address;

-- Update frame buffer flag: when am_new_frame is high, update frame buffer flag and current address

-- update_frame_buffer_flag : process(clk, n_reset)
-- begin
--     if n_reset = '0' then
--         frame_buffer_flag <= false;
--         current_offset <= (others => '0');
--     elsif rising_edge(clk) then
--         if am_new_frame = '1' then
--             frame_buffer_flag <= not frame_buffer_flag;
--             if frame_buffer_flag = true then
--                 current_offset <= to_unsigned(76800, 32);
--             else
--                 current_offset <= (others => '0');
--             end if;
--         end if;
--     end if;
-- end process update_frame_buffer_flag;

-------------------
-- Avalon Slave --
-------------------
process (clk, N_reset)
begin
    if n_reset = '0' then
	as_start <= '0';
	as_start_address <= (others => '0');
    elsif rising_edge(clk) then        
        -- if as_read = '1' then
        --     case as_address is
        --         when "00" => as_read_data <= (0 => enabled, others => '0');
		-- 			 when "01" => as_read_data <= addr????; 
		-- 			 when "10" => as_read_data <= mode????;
        --         when others => null;
        --     end case;
        -- elsif as_write = '1' then
        if as_write = '1' then
            case as_address is
                -- 0 : start_address
                when '0' =>
                    as_start_address <= as_write_data(31 downto 0);
                -- 1 : start
                when '1' =>
                    as_start <= as_write_data(0);
                when others => null;
            end case;
        end if;
    end if;
end process;

end architecture RTL;