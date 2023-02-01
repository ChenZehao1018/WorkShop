#include "msp.h"


/**
 * main.c
 */
int time (int a)
{
    //set the input proportionally to the duty cycle of PWM
    int b;
    b = (int) 33 + 33*a/0x3FFF;
    return b;
}

void ADC14_IRQHandler()
{
    //capture the ADC input and set the input proportionally to the duty cycle of PWM
    volatile int a = ADC14->MEM[0];
    TIMER_A0->CCR[1] = time(a);
}

void TA0_0_IRQHandler()
{
    if(TIMER_A0->CCTL[0] & TIMER_A_CCTLN_CCIFG==1)
    {
        //reset the flag of TimerA0 interrupt
        TIMER_A0->CCTL[0] &= ~TIMER_A_CCTLN_CCIFG;

        //enable ADC14 interrupt
        ADC14->IER0 = ADC14_IER0_IE0;

        //set ADC14, use A13, start sample and conversion
        ADC14->MCTL[0]=ADC14_MCTLN_INCH_13;
        ADC14->CTL0 = ADC14_CTL0_ON | ADC14_CTL0_ENC | ADC14_CTL0_SC;

        //offset
        ADC14->CTL0 &= ~ADC14_CTL0_SC;

        //wait for ADC14 interrupt
        volatile int i;
        for(i=0;i<10;i++);
    }
    else
    {
        TIMER_A0->CTL &= ~TIMER_A_CTL_IFG;
    }
}


void main(void)
{
	WDT_A->CTL = WDT_A_CTL_PW | WDT_A_CTL_HOLD;		// stop watchdog timer

	//set P4.0 as input, and use A13 function
    P4->SEL0=0;
    P4->SEL1=1;

    //set p2.4 as PWM output, and choose TA0.1 function
	P2->DIR = 0xFF;
    P2->SEL0=(1<<4);
    P2->SEL1=(0<<4);

    //set TimerA0,  use: up mode, ACLK, output mode 6 and enable interrupt.
	TIMER_A0->CTL =  TIMER_A_CTL_MC_1 | TIMER_A_CTL_SSEL__ACLK;
	TIMER_A0->CCR[0] = (int)32.768*20;
    TIMER_A0->CCR[1] = 0;
    TIMER_A0->CCTL[1] = TIMER_A_CCTLN_OUTMOD_6;
	TIMER_A0->CCTL[0] = TIMER_A_CCTLN_CCIE;

	//set port mapping controller, set P2.4 to output TimerA0 CCR1 compare out results.
    PMAPKEYID = PMAP_KEYID_VAL;
    P2MAP45 = PM_TA0CCR1A;
    PMAPKEYID = 0;

    //enable interrupt Timer A0, ADC14 and set their priority
    NVIC_EnableIRQ(ADC14_IRQn);
    NVIC_SetPriority(ADC14_IRQn,0);
    NVIC_EnableIRQ(TA0_0_IRQn);
    NVIC_SetPriority(TA0_0_IRQn,7);

    //wait for interrupt
	while(1);
}
