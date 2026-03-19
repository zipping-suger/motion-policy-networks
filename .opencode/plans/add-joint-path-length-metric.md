# Implementation Plan: Add Joint Path Length Metric (TrajOpt-style)

## Overview
Add a new metric "joint_path_length" to the Evaluator class that calculates the sum of squared displacements between consecutive joint configurations, following the TrajOpt formulation.

## Changes Required

### 1. Add calculation method to Evaluator class (mpinets/metrics.py)
Add new static method after `calculate_eff_path_lengths`:
```python
@staticmethod
def calculate_joint_path_length(trajectory: Trajectory) -> float:
    """
    Calculate the joint path length as sum of squared displacements between consecutive configurations
    following the TrajOpt formulation.
    
    :param trajectory Trajectory: The trajectory (sequence of 7DOF joint configurations)
    :rtype float: Joint path length value
    """
    configs = np.asarray(trajectory)
    return np.sum(np.square(np.diff(configs, axis=0)))
```

### 2. Integrate into trajectory evaluation (mpinets/metrics.py)
In the `evaluate_trajectory` method, after end effector path length calculation (lines 499-504):
```python
# Existing code
(
    eff_position_path_length,
    eff_orientation_path_length,
) = self.calculate_eff_path_lengths(trajectory)
add_metric("eff_position_path_length", eff_position_path_length)
add_metric("eff_orientation_path_length", eff_orientation_path_length)

# Add new code
joint_path_length = self.calculate_joint_path_length(trajectory)
add_metric("joint_path_length", joint_path_length)
```

### 3. Add metric aggregation (mpinets/metrics.py)
In the `metrics` static method:
1. After extracting eff path length arrays (lines 588-591), add:
```python
all_joint_path_lengths = np.asarray(group["joint_path_length"])
```
2. After calculating successful eff path lengths (lines 614-619), add:
```python
success_joint_path_lengths = all_joint_path_lengths[unskipped_successes]
joint_path_length = (
    np.mean(success_joint_path_lengths),
    np.std(success_joint_path_lengths),
)
```
3. Add to returned metrics dict (line 672):
```python
"joint_path_length": joint_path_length,
```

### 4. Add to print output (mpinets/metrics.py)
In the `print_metrics` method, after end effector orientation path length print (lines 709-713), add:
```python
print(
    "Average Joint Path Length (TrajOpt-style):"
    f" {metrics['joint_path_length'][0]:4.2f}"
    f" ± {metrics['joint_path_length'][1]:4.2f}"
)
```

### 5. Optional: Add to GUI printout (mpinets/metrics.py)
In the GUI debug print section (line 534), add:
```python
print(f"Joint Path Length: {joint_path_length}")
```

## Impact
- The new metric will be automatically calculated for all trajectories evaluated by both `calculate_metrics` and `visualize_results` functions in `run_inference.py`
- Metric will be displayed in both group and overall metrics output
- No changes required to `run_inference.py` as all integration is done within the Evaluator class
