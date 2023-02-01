library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity camera_module is
    port (
        clk     : in std_logic;
        n_reset : in std_logic;

        ------------------
        -- Avalon Slave --
        ------------------
        -- Avalon Slave <-> Avalon Bus
        as_address    : in std_logic;
        as_write      : in std_logic;
        as_write_data : in std_logic_vector(31 downto 0);

        -- Avalon Slave <-> Avalon Master
        -- as_start_address : out std_logic_vector(31 downto 0);

        -- Avalon Slave <-> Camera Interface
        -- as_start : out std_logic;

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

        -- Camera Interface <-> Camera
        pixel_clk   : in std_logic;
        line_valid  : in std_logic;
        frame_valid : in std_logic;
        cam_data    : in std_logic_vector(11 downto 0);
        cam_reset_n : out std_logic;
        xclkin      : out std_logic
    );
end camera_module;

architecture rtl of camera_module is
    component camera_interface is
        port (
            clk              : in std_logic;
            n_reset          : in std_logic;
            ci_RGB_pixel_out : out std_logic_vector(15 downto 0);
            ci_new_data      : out std_logic;
            ci_data_ack      : in std_logic;
            ci_new_frame     : out std_logic;
            ci_start         : in std_logic;
            pixel_clk        : in std_logic;
            line_valid       : in std_logic;
            frame_valid      : in std_logic;
            cam_data         : in std_logic_vector(11 downto 0);
            cam_reset_n      : out std_logic;
            xclkin           : out std_logic
        );
    end component camera_interface;

    component camera_avalon_manager is
        port (
            clk               : in std_logic;
            n_reset           : in std_logic;
            as_address        : in std_logic;
            as_write          : in std_logic;
            as_write_data     : in std_logic_vector(31 downto 0);
            as_start          : out std_logic;
            am_address        : out std_logic_vector(31 downto 0);
            am_byte_enable    : out std_logic_vector(3 downto 0);
            am_burst_count    : out std_logic_vector(3 downto 0); -- modify graph
            am_write          : out std_logic;
            am_write_data     : out std_logic_vector(31 downto 0);
            am_wait_request   : in std_logic;
            am_pixel_transfer : in std_logic_vector(15 downto 0);
            am_new_data       : in std_logic;
            am_data_ack       : out std_logic;
            am_new_frame      : in std_logic
        );
    end component camera_avalon_manager;

    signal ci_RGB_pixel_out : std_logic_vector(15 downto 0);
    signal ci_new_data      : std_logic;
    signal ci_data_ack      : std_logic;
    signal ci_new_frame     : std_logic;

    signal as_start : std_logic;
begin
    ci: camera_interface port map(
        clk              => clk,           
        n_reset          => n_reset, 
        ci_RGB_pixel_out => ci_RGB_pixel_out,
        ci_new_data      => ci_new_data,
        ci_data_ack      => ci_data_ack,
        ci_new_frame     => ci_new_frame,
        ci_start         => as_start,
        pixel_clk        => pixel_clk,
        line_valid       => line_valid, 
        frame_valid      => frame_valid,
        cam_data         => cam_data,
        cam_reset_n      => cam_reset_n,
        xclkin           => xclkin
    );

    cm: camera_avalon_manager port map(
        clk               => clk,
        n_reset           => n_reset,
        as_address        => as_address,
        as_write          => as_write,
        as_write_data     => as_write_data,
        as_start          => as_start,
        am_address        => am_address,
        am_byte_enable    => am_byte_enable,
        am_burst_count    => am_burst_count,
        am_write          => am_write,
        am_write_data     => am_write_data,
        am_wait_request   => am_wait_request,
        am_pixel_transfer => ci_RGB_pixel_out,
        am_new_data       => ci_new_data,
        am_data_ack       => ci_data_ack,
        am_new_frame      => ci_new_frame
    );
end;