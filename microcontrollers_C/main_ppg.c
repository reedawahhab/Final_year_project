/* =============================================================================
 * PPG FIRMWARE  -  20 Hz output for Python viewer
 * Based on Sasai et al., Sensors 2019
 * =============================================================================
 *
 * Main changes from your previous draft:
 *   1) TIM2 used as a free-running 1 MHz timer for accurate microsecond timing
 *   2) ADC reads are averaged (8 samples) for both t1 and t2
 *   3) INTEGRATE_MS kept at 12 ms
 *   4) Cycle paced precisely to 50 ms (20 Hz)
 *
 * Notes:
 *   - Output packet is unchanged:
 *       0xAA 0x55 0x01 0x00 [uint16 little-endian sample]
 *   - Viewer expects output centred around 2048
 *   - If signal clips too often, reduce INTEGRATE_MS to 10
 *   - If signal is too weak, try 14 ms
 * =============================================================================
 */

#include "main.h"
#include <stdio.h>
#include <string.h>

/* ────────────────────────────────────────────────────────────────────────── */
/* Peripheral handles                                                        */
/* ────────────────────────────────────────────────────────────────────────── */
ADC_HandleTypeDef  hadc1;
DMA_HandleTypeDef  hdma_adc1;
TIM_HandleTypeDef  htim2;
UART_HandleTypeDef huart2;

/* ────────────────────────────────────────────────────────────────────────── */
/* Function prototypes                                                       */
/* ────────────────────────────────────────────────────────────────────────── */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM2_Init(void);

/* ────────────────────────────────────────────────────────────────────────── */
/* Timing configuration                                                      */
/* ────────────────────────────────────────────────────────────────────────── */

#define CYCLE_US      (50000u)


#define COOLDOWN_US      (10u)

#define SETTLE_US       (100u)

#define INTEGRATE_US (3000u) //max 100 decrease more -> more chance to get t2-t1

/* Number of ADC samples to average for t1 and t2 */
#define ADC_AVG_N         1u
char result_buffer[100];

/* ────────────────────────────────────────────────────────────────────────── */
/* GPIO helpers                                                              */
/* ────────────────────────────────────────────────────────────────────────── */
#define LED_ON()    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET)
#define LED_OFF()   HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET)

/* TS5A3166 / TS5A3167 style control:
 * IN = HIGH -> switch CLOSED
 * IN = LOW  -> switch OPEN
 */
#define SW_CLOSE()  do {                                                      \
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_SET);                       \
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, GPIO_PIN_SET);                       \
} while (0)

#define SW_OPEN()   do {                                                      \
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, GPIO_PIN_RESET);                     \
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, GPIO_PIN_RESET);                     \
} while (0)

/* ────────────────────────────────────────────────────────────────────────── */
/* TIM2 helpers: free-running at 1 MHz => 1 tick = 1 us                      */
/* ────────────────────────────────────────────────────────────────────────── */
static inline uint32_t tim2_us(void)
{
    return __HAL_TIM_GET_COUNTER(&htim2);
}

static void delay_us_tim2(uint32_t us)
{
    uint32_t start = tim2_us();
    while ((uint32_t)(tim2_us() - start) < us) {
        /* wait */
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* ADC helpers                                                               */
/* ────────────────────────────────────────────────────────────────────────── */
static inline uint16_t adc_read_once(void)
{
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 2);
    return (uint16_t)HAL_ADC_GetValue(&hadc1);
}

static uint16_t adc_read_avg(uint8_t n)
{
    uint32_t sum = 0;
    for (uint8_t i = 0; i < n; i++) {
        sum += adc_read_once();
    }
    return (uint16_t)(sum / n);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* UART packet                                                                */
/* ────────────────────────────────────────────────────────────────────────── */
//static void send_packet(uint16_t value)
static void send_packet(uint16_t t1, uint16_t t2)
{
//    uint8_t buf[6] = {
//        0xAAu, 0x55u,
//        0x01u, 0x00u,
//        (uint8_t)(value & 0xFFu),
//        (uint8_t)((value >> 8) & 0xFFu)
//    };
//    HAL_UART_Transmit(&huart2, buf, sizeof(buf), HAL_MAX_DELAY);
	//snprintf(result_buffer, sizeof(result_buffer), "%u\r\n", value);
	int32_t diff = (int32_t)t2 - (int32_t)t1;
	snprintf(result_buffer, sizeof(result_buffer),
	         "%u,%u,%ld\r\n", t1, t2, (long)diff);
	HAL_UART_Transmit(&huart2, (uint8_t *)result_buffer, strlen(result_buffer),HAL_MAX_DELAY);

}


/* ────────────────────────────────────────────────────────────────────────── */
/* Main                                                                      */
/* ────────────────────────────────────────────────────────────────────────── */
int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_USART2_UART_Init();
    MX_ADC1_Init();
    MX_TIM2_Init();

    HAL_TIM_Base_Start(&htim2);

    LED_OFF();
    SW_CLOSE();
    HAL_Delay(20);   /* allow supply / analog front-end to settle */



    while (1)
    {
        uint32_t cycle_start_us = tim2_us();


        SW_OPEN();
        delay_us_tim2(SETTLE_US);

        uint16_t t1 = adc_read_once();
        LED_ON();


        delay_us_tim2(INTEGRATE_US);
        uint16_t t2 = adc_read_once();
        LED_OFF();



        SW_CLOSE();
        delay_us_tim2(COOLDOWN_US);





        /* Step 6: CDS subtraction and centre shift */
        int32_t diff = (int32_t)t2 - (int32_t)t1;
        int32_t out  = diff + 2048;

        if (out < 0)    out = 0;
        if (out > 4095) out = 4095;

        //send_packet(out);
        send_packet(t1,t2);

        /* Step 7: pace to exactly 50 ms total cycle */
        while ((uint32_t)(tim2_us() - cycle_start_us) < CYCLE_US) {
            /* wait */
        }
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Clock config                                                              */
/* ────────────────────────────────────────────────────────────────────────── */
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef osc = {0};
    RCC_ClkInitTypeDef clk = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

    osc.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
    osc.HSIState            = RCC_HSI_ON;
    osc.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    osc.PLL.PLLState        = RCC_PLL_ON;
    osc.PLL.PLLSource       = RCC_PLLSOURCE_HSI;
    osc.PLL.PLLM            = 16;
    osc.PLL.PLLN            = 336;
    osc.PLL.PLLP            = RCC_PLLP_DIV4;
    osc.PLL.PLLQ            = 7;
    if (HAL_RCC_OscConfig(&osc) != HAL_OK) Error_Handler();

    clk.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                       | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider = RCC_HCLK_DIV2;
    clk.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_2) != HAL_OK) Error_Handler();
}

/* ────────────────────────────────────────────────────────────────────────── */
/* ADC1 init                                                                 */
/* ────────────────────────────────────────────────────────────────────────── */
static void MX_ADC1_Init(void)
{
    ADC_ChannelConfTypeDef s = {0};

    hadc1.Instance                   = ADC1;
    hadc1.Init.ClockPrescaler        = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution            = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode          = DISABLE;
    hadc1.Init.ContinuousConvMode    = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConvEdge  = ADC_EXTERNALTRIGCONVEDGE_NONE;
    hadc1.Init.ExternalTrigConv      = ADC_SOFTWARE_START;
    hadc1.Init.DataAlign             = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion       = 1;
    hadc1.Init.DMAContinuousRequests = DISABLE;
    hadc1.Init.EOCSelection          = ADC_EOC_SINGLE_CONV;
    if (HAL_ADC_Init(&hadc1) != HAL_OK) Error_Handler();

    s.Channel      = ADC_CHANNEL_0;
    s.Rank         = 1;
    s.SamplingTime = ADC_SAMPLETIME_84CYCLES;
    if (HAL_ADC_ConfigChannel(&hadc1, &s) != HAL_OK) Error_Handler();
}

/* ────────────────────────────────────────────────────────────────────────── */
/* TIM2 init: 1 MHz free-running counter                                     */
/* ────────────────────────────────────────────────────────────────────────── */
static void MX_TIM2_Init(void)
{
    TIM_ClockConfigTypeDef  cc = {0};
    TIM_MasterConfigTypeDef mc = {0};

    htim2.Instance               = TIM2;
    htim2.Init.Prescaler         = 83;                  /* 84 MHz / 84 = 1 MHz */
    htim2.Init.CounterMode       = TIM_COUNTERMODE_UP;
    htim2.Init.Period            = 0xFFFFFFFFu;         /* free-running */
    htim2.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim2) != HAL_OK) Error_Handler();

    cc.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
    if (HAL_TIM_ConfigClockSource(&htim2, &cc) != HAL_OK) Error_Handler();

    mc.MasterOutputTrigger = TIM_TRGO_RESET;
    mc.MasterSlaveMode     = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &mc) != HAL_OK) Error_Handler();
}

/* ────────────────────────────────────────────────────────────────────────── */
/* USART2 init                                                               */
/* ────────────────────────────────────────────────────────────────────────── */
static void MX_USART2_UART_Init(void)
{
    huart2.Instance          = USART2;
    huart2.Init.BaudRate     = 921600;
    huart2.Init.WordLength   = UART_WORDLENGTH_8B;
    huart2.Init.StopBits     = UART_STOPBITS_1;
    huart2.Init.Parity       = UART_PARITY_NONE;
    huart2.Init.Mode         = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    if (HAL_UART_Init(&huart2) != HAL_OK) Error_Handler();
}

/* ────────────────────────────────────────────────────────────────────────── */
/* DMA init                                                                  */
/* ────────────────────────────────────────────────────────────────────────── */
static void MX_DMA_Init(void)
{
    __HAL_RCC_DMA2_CLK_ENABLE();
    HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* GPIO init                                                                 */
/* ────────────────────────────────────────────────────────────────────────── */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef g = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* PB4 = SW2, PB5 = SW1, PB6 = LED */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6, GPIO_PIN_RESET);

    g.Pin   =  GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6;
    g.Mode  = GPIO_MODE_OUTPUT_PP;
    g.Pull  = GPIO_NOPULL;
    g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &g);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Error handler                                                             */
/* ────────────────────────────────────────────────────────────────────────── */
void Error_Handler(void)
{
    __disable_irq();
    while (1) {
    }
}
