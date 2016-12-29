# Tech primer oppenheimer

## Supply chain and key components

### Electronics industry supply chain
---
1. **Foundry and A&T** (晶元代工/半导体芯片生产线/封装测试） 
	- Foundry and Assembly and Test
	- Perform manufacturing of semiconductor devices
	- Examples include TSMC（台积电）SMIC (中芯国际) UMC ASE and Amkor
	- **_Upstream_**
2. **Semiconductor vendor**
	- Perform chip design and marketing
	- _Some manufacture in-house (Intel), some uses foundry and A&Ts (Qualcomm)_
	- Sell directly to OEMs, ODMs or EMS and sell to distributors
	- Examples include Intel, TI, Qualcomm
	- **_Upstream_**
3. **Semiconductor distributors**
	- Carry inventory to help smooth supply chain
	- handle import logistics to simplify international shipments
	- reach smaller customers that the chip vendor cannot service directly
	- Examples include Avnet and Arros
4. **Electronics Manufacturing Services (EMS)** 
	- Perform manufacturing on behalf of OEMs
	- Sometimes also handle procurement
	- Examples include Flextronics and Jabil
	- **Upstream**
5. **Original Device Manufacturer (ODM)**
	- Simliar to EMS but one step further
	- Take over design, procurement and manufacture
	- Contractor does not own IP over product
	- Can partner with OEM or SPs who perform marketing and distribution
	- Examples include Hon Hai, BenQ, Compal and Quanta
6. **Original Equipment Manufacturer (OEM)**
	- Electronics providers who design and market branded products to end customers and SPs
	- Various vertical integration models
		- Some do everything from chip design and manufacturing all the way to direct sales to customers
		- Others focus on marketing and some aspects of design
	- Examples include HP, Dell, Apple, Cisco, Nokia, Sony, Samsung etc
	- **Midstream**
7. **Equipment distributor**
	- Help OEM reach customers 
	- Service individual retailers and selling directly to customers
	- Examples include CDW and Ingram Micro
	- **Downstream**
8. **Electronics retailer**
	- Best Buy Walmart etc
	- **Downstream**


###Lithography and Moore's Law 光刻和摩尔定律
---
**Lithography** is a general term used for the set of processes that transfer the transistor array and interconnect design from the mask onto the silicon wafer. It describes the level of “smallness” the manufacturing process can achieve, which determines the size of the transistors on the die. Smaller transistors can lead to:

- Faster speed as electrons move quicker in between
- Cheaper devices as more transistors per square inch lead to smaller die size
- Lower power consumption as less electricity is needed

**Moore's law** is primarily driven by a  dvancement towards finer lithography, which enables smaller and smaller transistors

- Number of transistors on a chip **_double about every 24 month_**
- First observed by Intel co-founder Moore
- In addition to the building of transistor density, each new lithography generation usually brings 0.7x minimum feature scaling, 1.5x faster transitor switching speed, reduced chip power and reduced chip cost
- [ ] **any chance to replicate chart with my own data?**


###Manufacturing segments
---

1. **_Integrated Device Manufactuers (IDM)_**
	- 自己设计 自己生产
	- Typically mature vendors dealing with high volume products
	- Have operating and cyclical leverage
	- Examples include Intel, TI, Samsung, Toshiba
	- IDMs enjoy high margins in the upward cycle as their fixed costs do not change, but suffer low margins in the trough of the cycle as capacity goes unused
	- This effect should not be underestimated, as fixed costs (including depreciation and fixed material and labor costs) can run as high as 80% of semiconductor manufacturing costs
2. **_Fabless_**
	- 自己设计 代工生产
	- More margin and stability as well as low capex
	- Examples include Broadcom, Qualcomm, NVIDIA, Xilinx, Marvell
	- Fabless companies can gain share vs. IDMs in downturn
		- wafer costs for them will decline while unit costs for IDMs rise with declining output
	- Fabless will lose share at the peak of the cycle (because of the opposite effect)
3. **_Foundries and Assembly and Testers_**
	- 不设计 只生产
	- Very high capex (tens of billions US$)
	- Examples cinlude TSMC, Global Foundries, UMC, Samssung, SMIC

###Foundries strategies 晶元代工策略
---

Rising cost of semiconductor lab have made outsourcing more cost-effective. Major foundries have **OUTPACED** semiconductor industry revenue growth for the past sevel years.

- 半导体行业2015年industry revenue **US$333 billion**
- 晶圆代工行业2015年industry revenue **US$48.8 billion**, 15% of total semiconductor industry
- 2015年晶圆代工行业销售额分布
	- TSMC (台积电) 54.3%, up from 47% in 2010
	- Global Foundries 9.6%, down from 12% in 2010
	- UMC 9.3%, down from 14% in 2010
	- Samsung 5.3%
	- SMIC (中芯国际) 4.6%

###Assembly and test strategies 封装测试策略
---

Trend in outsourcing also benefited back-end (A&T) providers. IDMs often keep back-end production in-house but fabless companies ususally outsource.

- 半导体行业2015年industry revenue **US$333 billion**
- 晶圆代工行业2010年industry revenue **US$22 billion**
- _2010_年封装测试行业销售额分布：
	- ASE Group 16%
	- Amkor 12%
	- Siliconware Precision 9%
	- STATs ChipPac 7%
	- PowerTech 5%

##Industry Fundamentals

### Industry fundamentals - demand
---

- Electronics industries totaled US$1.4 trillion in 2010
	- 28% data processing equipment
		- PCs 42%
		- Diskdrives 10%
	- 27% communication equipment 
		- Handsets 55%
		- Telecom infrastructure 17%
	- 19% consumer electronics 
		- Appliances 39%
		- Other consumer 26%
	- 26% others
		- Industrial 48%
		- Automobile 26%
		- Aerospace 26%

- Semiconductor
	- US$300 billion market in 2010
	- 22% of total Electronics market
	- chipmakers increasingly include more system IP into their device. 
		- Chip prices increasingly include software components
	- Chipmakers mostly have pricing power, OEMs have been subject to tough competition
	- OEM increasingly use outsourcing / inventory management to control costs
	- OEM is also charging service fee in addition to manufacturing

### Industry fundamentals - supply
---

- Supply mostly dependent on wafer fab capacities
	- A wafer fab will need multi billion dollars to build
- Very challenging to match demand / supply and produce ahead of time
- Semiconductor in general is a supply-driven industry with average 4-6 years cycle

#### Evolving cycle
- Since 2001 bust, the growing use of foundries by fabless and IDMs has removed leverage	from semiconductor business models
- Diviersifcation of end market lessens dependence on PC cycles
- Implications:
	- New cyles are more based on inventory than supply
	- Shorter cycles with shallower ups and downs
	- Gross margin less relevant
	- Memory cycles more independent from the broader semiconductor cycle
	- As long as macro economy is healthy, even in weak fundamental period revenue can grow.
	- 开始集体外包给foundries (台积电)
- **过去是完全生产供给导向，现在是库存供给导向**


### Industry fundamentals - sales 
---

- Annual sale of US$300 billion in 2010. 
- Contraction in 2008 and 2009 led to revenue falling below US$250 billion
- Structural changes during that recover have **_replaced the supply-driven cycle with inventory-driven cycle_**.
- More consistent and manageable growth rates observed since

### Industry fundamentals - ASP and shipping units
---

- Cycles become shorter (2-3 years)
- in downturns, shipping units are likely the same but ASPs will fluctuate

### Industry fundamentals - Capacity utlisation
- Key measure of supply: **_Capacitiy utilisation_**
	- It also provides the link between units and ASPs
	- Utilisation is tracked for front end (wafer processing) / assembly and testing, but front end utilisation is most important because:
	- Cycle times for the front end are longest
	- Most difficult to add and remove front end capacity
	- front fabs most expensive
	- In high demand season, both volume and price are upward lifted
	- In low demand season, both volume and price and downward lifted

### Industry fundamentals - Capital Spending
---

- Estimated to be 23% - 25% of semiconductor revenues
- Increased outsource has shifted supply risk to the foundries whoare a lot more prudent and rational about spending because they have industry-wide perspective
- Helped shift supply-driven cycles to inventory-driven cycles
- Helped bring stability with the semiconductor capital equipment industry, which historically has been **_more drmatically cyclical_** than the semiconductor industry itself.


### Industry fundamentals - Gross margin
---

- Gross margin for the industry is in mid-40% range
- Outsourcing have stablized margin volatility since the 2000 bubble
	- Depreciation accounts for a major portion of COGs
	- Gross margin tend to rise and fall with capacity utilisation rates
	- Outsourcing really helped margin stablisation, even in the 2008-2009 correction, industry gross margin remain at just below 40%

### Industry fundamentals - Operating margins
---

- Opearting margin historically averaged under 15%
	- R&D + SG&A accounts for 25% - 30% of revenue
	- Dropped negative in 2009 but recovered quickly
	- Even in downturn companies remain heavily invested in R&D

### Industry fundamentals - Receivables & Inventories

- Receivable turnover days average 40 days, dropped from ~50 days in early 2000
- Inventories have risen over the past few years from 75 days to 78 days
	- OEMS are reluctant to hold inventories
	- rise in outsourcing allows vendors to hold more inventory with less risk
- More important inventory is the inventory downstream in OEMs, EMS and distributors. This is the **_KEY variable_** driving the **_cyclicality_** of the industry.
	- When inventories run too low, OEMS tend to double-book in order to guarantee supply to meet demand
	- When inventories run too low, OEMs tend to push out orders.
	- PC and Server oem inventories are continuously trending down - near 15-20 days from 50 days in 2000
	- HDD inventory days also tend to be relatively low - near 25 days.

### Industry fundamentals - debt
---

- debt to equity mostly around 20%
	- primarily concentrated in the hands of IDMs and oursourcers.
	- fluctuations tend to be driven by equity instead of debt
	- buy-backs and debt recapitalization was prevalent in 2007 resulting in a higher debt to equity

## Market Segments

### Market Segments - Device types
- Analog / 模拟信号
	- Deal directly with outside feedback (voltages / currents / charge) . 
	- Transitors designed to manipulate voltages and currents
	- Well suited to process real-world signals, electronic patterns are used to directly represent the original signal
	- 通过电压、电流变化直接对应外界真实信号反馈
- Digital / 数字信号
	- 以0和1代表外界电压/电流信号是否存在
	- 大量应用于进一步处理计算，可将模拟信号转为数字信号后进行压缩、差错等操作
- Mixed Signal / 混合信号
	- 混合模拟和数字信号 

- What a device IS: 

	- Discretes, Optoelectronics and Sensors
		- All non-integrated circuit semiconductor devices
		- Discrete is a single transistor in package
		- Sensors is a discrete device that measures real-world inputs
		- Optoelectronics produce or measure light
		- 非集成的单一单元，晶体管，感应器，发光器
	- Analog
		- Process real-world signals using electroic voltage patterns that represent the signal
		- Include SLIC (standard linear components)
		- ASSPs (application-specific analog ICs)
		- 处理真实世界信号，包括单一原件和特定原件
	- Microcomponents
		- Digital processors 
		- include microprocessors (MPUs) 大脑
		- microcontrollers (MCUs) 计算
		- digital signal processors (DSPs) 处理信号
		- 微信号数字处理器
		- 需将模拟信号转成数字信号后才能处理
	- Logic
		- Non microcomponent digital logic
		- ASIC (custom logic)
		- ASSPs
		- FPGA (programmable logic)
		- display drivers
	- Memory
		- store data either in RAM or ROM

- What a device DOES: 

	- Analog processing
		- 处理真实世界信号
	- Digital processing
		- 处理数字信号，编译信号，压缩解压缩
	- System procesors
		- 系统层面控制电信号，包括所有的微处理器
	- Memory 
		- 储存
	- System enabling device 
		- 电源、开关、时钟、感应器等等

| End Market application  | Analog Processors | Digital Processors | System Processors | Memory | System enabling devices
|:------------- | :---:| :---: | :---: | :---:| :---: |
| **Computing** |
| Desktop PC | Graphic Interface, sound interface, Ethernet PHY | Chipset, Graphic Processor, Sound Processor | Microprocessor | DRAM ROM Flash | Power ICs, clocks, switches, discretes |
| Notebook PC | Graphic/Sound interface, Ethernet PHY, WLAN/Bluetooth | Chipset, graphics processor, sound processor, WLAN/Bluetooth baseband / MAC | Microprocessor | DRAM, ROM, Flash | Power ICs, clocks, switches, discretes
| HDD | Read channel, preamp, motor driver | Servo asic, hard disk controller | Microcontroller | DRAM, SRAM, flash | Magnetic sensors, power ICs, clocks, switches, discretes
| **Communications** | 
| Wireless handset | Power amp, radio (RF/IF), audio codec, Bluetooth radio, analog baseband | Digital baseband (DSP), Bluetooth baseband/MAC, Camera image processor, Multimedia processor | Applications processor (DSP embedded) | Flash, SRAM, low-power DRAM | Sensors, power ICs, switches, clocks, discretes, LCD drivers |
| Ethernet switch | Ethernet PHY | Ethernet Mac, switch chipset, packet processor, programmable logic | MIPs/Powerpc Processors | SRAM / DRAM / ROM | Power ICs, clocks, switches, discretes


### Key semiconductor Competitors
---

####Top 10 
- Intel (US) 
- Samsung (Korea)
- Toshiba (Japan)
- Texas instruments (US)
- Renesas (Japan)
- Hynix (Korea)
- STMicroelectronics(Switzerland)
- Micro Technology (Japan)
- Qualcomm (US)
- Infineon / Qimonda (Germany)

### Semiconductor market
---

- 2010 market sales of US$298 billion
- 2015 market sales of US$335 billion, roughly 4% CAGR
- Most sub-segments are estimated to grow at 4% - 6% annually
- industry growth at 5% annualy