library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity demosaicking is
    port (
        G1        : in std_logic_vector (11 downto 0);
        R         : in std_logic_vector (11 downto 0);
        B         : in std_logic_vector (11 downto 0);
        G2        : in std_logic_vector (11 downto 0);
        RGB_pixel : out std_logic_vector (15 downto 0)
    );
end demosaicking;

architecture rtl of demosaicking is
    signal G_sum : std_logic_vector (11 downto 0);
    signal G_avg : std_logic_vector (11 downto 0);
begin
    G_sum <= std_logic_vector((unsigned(G1) + unsigned(G2)));
    G_avg <= '0' & G_sum(11 downto 1);
    RGB_pixel <=  R(4 downto 0) & G_avg(5 downto 0) & B(4 downto 0);
end;