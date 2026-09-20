# Locomotion Algorithm
# 1. Baseline
初期候補:
- LIPM
- Capture Point
- Foot Placement
- MPC
- Hybrid MPC for Knee Lock
# 2. State
- CoM position / velocity
- body attitude
- support foot
- contact state
- hip/thigh actuator state
- knee lock state
- ankle spring deflection
# 3. Output
- footstep target
- body target
- hip/thigh muscle target
- knee lock timing
# 4. Sequence
```text
Estimate state
→ Determine support phase
→ Compute capture / step target
→ Generate body trajectory
→ Allocate muscle group targets
→ Determine knee lock/release
→ Apply safety / power limit
```
