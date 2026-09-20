# PWR-DTL-010 Interface / Configuration
# 1. C++ Interface Concept
```cpp
enum class PowerMode
{
    Off,
    Starting,
    On,
    Warning,
    Limited,
    Fault,
    Shutdown
};
struct PowerBranchState
{
    std::string id;
    bool enabled;
    float voltage;
    float current;
    float power;
    float currentLimit;
    int priority;
    bool fault;
};
struct PowerState
{
    PowerMode mode;
    float batteryVoltage;
    float batteryCurrent;
    float soc;
    float soh;
    float totalPower;
    float availablePower;
    std::vector<PowerBranchState> branches;
};
class IPowerSystem
{
public:
    virtual ~IPowerSystem() =default;
    virtual PowerState getState() const =0;
    virtual bool setBranchEnabled(const std::string& id,bool enabled) =0;
    virtual bool setBranchCurrentLimit(const std::string& id,float ampere) =0;
    virtual void requestSafeShutdown() =0;
};
```
# 2. Configuration
```text
PowerConfig
├── battery_nominal_voltage
├── battery_capacity_ah
├── usable_ratio
├── nominal_power_limit
├── intense_power_limit
├── short_peak_limit
├── rail_limits[]
├── branch_limits[]
├── thermal_limits[]
└── shutdown_timing[]
```
# 3. Branch IDs
```text
PWR-NECK
PWR-UPPER-L
PWR-UPPER-R
PWR-WAIST-L
PWR-WAIST-R
PWR-LOWER-L
PWR-LOWER-R
PWR-LOGIC
PWR-AUX
```
# 4. Control Interface
Controlは必要PowerをRequestし、Power Systemは許容値を返す。
```text
PowerRequest(group, requested_power)
→ PowerGrant(group, allowed_power, current_limit, limited)
```
