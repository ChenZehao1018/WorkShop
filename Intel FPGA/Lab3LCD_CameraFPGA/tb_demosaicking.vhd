library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_demosaicking is
end tb_demosaicking;

architecture test of tb_demosaicking is
    signal G1        : std_logic_vector(11 downto 0);
    signal R         : std_logic_vector(11 downto 0);
    signal B         : std_logic_vector(11 downto 0);
    signal G2        : std_logic_vector(11 downto 0);
    signal RGB_pixel : std_logic_vector(15 downto 0);
begin
    dut: entity work.demosaicking
        port map(
            G1        => G1,
            R         => R,
            B         => B,
            G2        => G2,
            RGB_pixel => RGB_pixel
        );

    simulation: process
    begin
        G1 <= std_logic_vector(to_unsigned(1, G1'length));
        R <= std_logic_vector(to_unsigned(1, R'length));
        B <= std_logic_vector(to_unsigned(1, B'length));
        G2 <= std_logic_vector(to_unsigned(1, G2'length));
        wait for 20 ns;

        G1 <= std_logic_vector(to_unsigned(1, G1'length));
        R <= std_logic_vector(to_unsigned(1, R'length));
        B <= std_logic_vector(to_unsigned(1, B'length));
        G2 <= std_logic_vector(to_unsigned(3, G2'length));
        wait for 20 ns;

        wait;
    end process simulation;

end architecture test;
