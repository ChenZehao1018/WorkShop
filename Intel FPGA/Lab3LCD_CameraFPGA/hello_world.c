/*
 * "Hello World" example.
 *
 * This example prints 'Hello from Nios II' to the STDOUT stream. It runs on
 * the Nios II 'standard', 'full_featured', 'fast', and 'low_cost' example
 * designs. It runs with or without the MicroC/OS-II RTOS and requires a STDOUT
 * device in your system's hardware.
 * The memory footprint of this hosted application is ~69 kbytes by default
 * using the standard reference design.
 *
 * For a reduced footprint version of this template, and an explanation of how
 * to reduce the memory footprint for a given application, see the
 * "small_hello_world" template.
 *
 */
#include <inttypes.h>

#define CAMERA_CONTROLLER_0_BASE 0x10000808


#include <stdio.h>
#include <system.h>
#include <io.h>
#include <altera_avalon_pio_regs.h>
#include <stdint.h>

//constant
#define AS_ADDR_WIDTH 1
#define REG_START_ADDRESS 4
#define REG_BURST_COUNT 0x8
#define REG_DATA_LENTH 240
#define ONE_MB 320 * 240


// Register addresses in Avalon Slave
#define ADDR_START_ADDRESS      (0*AS_ADDR_WIDTH ) // 000 ok
#define ADDR_DATA_LENTH      (1*AS_ADDR_WIDTH ) // 001 ok
#define ADDR_BURST_COUNT      (2*AS_ADDR_WIDTH ) // 010 ok
#define ADDR_LCD_EN      (3*AS_ADDR_WIDTH ) // 011  ok
#define ADDR_LCD_CMD_DATA             (4*AS_ADDR_WIDTH ) // 100  ok
#define ADDR_CHOOSE_CMD_DATA    (5*AS_ADDR_WIDTH ) // 101 (01:cmd;10:data)  ok
#define ADDR_AM_EN             (6*AS_ADDR_WIDTH ) // 110



int main()
{
	printf("Hello from Nios II!\n");
  	//picin();
  	uint32_t megabyte_count = 0;
  	for (uint32_t i = 0; i < 320*120*16; i += sizeof(uint32_t)){

    	      // Print progress through 256 MB memory available through address span expander
        	  if ((i % ONE_MB) == 0) {
            	  printf("megabyte_count = %d \n", megabyte_count);
              	megabyte_count++;
         	 }

          	uint32_t addr = HPS_0_BRIDGES_BASE + i;

	          // Write through address span expander
    	      uint32_t writedata = i;
        	  IOWR_32DIRECT(addr, 0, writedata);

          	// Read through address span expander
         	 uint32_t readdata = IORD_32DIRECT(addr, 0);

    }
	IOWR_32DIRECT(CAMERA_CONTROLLER_0_BASE, 0, 0);
	IOWR_32DIRECT(CAMERA_CONTROLLER_0_BASE, 1, 1);

  	printf("Initializing LCD...\n");
  	LCD_write_registers();
  	LCD_turn_on();
  	LCD_configure();
  	printf("Initialization completed.\n");
  	LCD_display();
	//turn on the master
  	IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_AM_EN, 1);
  	usleep(1000000000000000000);
}




//write data
void LCD_WR_DATA(uint32_t data) {
	// RegCommandOrData = 10b to write data
	IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_CHOOSE_CMD_DATA, 0x00000002);
    // Write to RegCommandData
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_LCD_CMD_DATA, data);
    usleep(1);
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_CHOOSE_CMD_DATA, 0x00000000);
}

//write cmd
void LCD_WR_CMD(uint32_t command) {
	// RegCommandOrData = 01b to write cmd
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_CHOOSE_CMD_DATA, 0x00000001);
    // Write to RegCommandData
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_LCD_CMD_DATA, command);
    usleep(1);
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_CHOOSE_CMD_DATA, 0x00000000);
}

//turn on lcd
void LCD_turn_on(void) {
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_LCD_EN, 1);
}

//turn off lcd
void LCD_turn_off(void) {
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_LCD_EN, 0);
}

//write to register
void LCD_write_registers(void) {
	printf("Writing to registers...\n");
    // Provide start address (RegStartAddress)
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_START_ADDRESS, REG_START_ADDRESS);
	printf("RegStartAddress = %d \n", IORD_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_START_ADDRESS));
	// Provide data lenth (RegDataLenth)
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_DATA_LENTH, REG_DATA_LENTH);
    printf("RegDataLenth = %d \n", IORD_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_DATA_LENTH));
    // Provide number of bursts per transfer (RegBurstCount)
    IOWR_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_BURST_COUNT, 240);
    printf("RegBurstCount = %d \n", IORD_32DIRECT(LCD_CONTROLLER_TOP_0_BASE, ADDR_BURST_COUNT));
}
//configure LCD
void LCD_configure(void) {


	LCD_WR_CMD(0x0011); //Exit Sleep
	//Delay_Ms(100);


	LCD_WR_CMD(0x0029); //display on



	LCD_WR_CMD(0x0036); // Memory Access Control
		 LCD_WR_DATA(0x0020); // Row/Column Exchange




	LCD_WR_CMD(0x002A); // Column Address Set
		 LCD_WR_DATA(0x0000);
		 LCD_WR_DATA(0x0000);
		 LCD_WR_DATA(0x0001);
		 LCD_WR_DATA(0x003F); //if MADCTL's B5 = 0, If B5=1, use LCD_WR_DATA(0x0013F);

	LCD_WR_CMD(0x002B); // Page Address Set
		 LCD_WR_DATA(0x0000);
		 LCD_WR_DATA(0x0000);
		 LCD_WR_DATA(0x0000);
		 LCD_WR_DATA(0x00EF); // or 0x00EF is MADCTL's B5=1

	LCD_WR_CMD(0x003A); // COLMOD: Pixel Format Set
	 	 LCD_WR_DATA(0x0055);

	LCD_WR_CMD(0x00f6); // Interface Control
		 LCD_WR_DATA(0x0001); // When the transfer number of data exceeds ( C-SC+1)*(EP-SP+1), the column and page number will be reset, and the exceeding data will be written into the following column and page.
		 LCD_WR_DATA(0x0030); // expand 16 bits data to 18bits frame : ¡°1¡± is inputted to LSB
		 LCD_WR_DATA(0x0000); // normal data transfert mode; MSB 1st
}

void LCD_display(void) {

    printf("Displaying image.\n");

    // 0x2C Display Command
    LCD_WR_CMD(0x0000002C);
}

void Delay_Ms(alt_u16 count_ms)
{
    while(count_ms--)
    {
        usleep(1000);
    }
}

