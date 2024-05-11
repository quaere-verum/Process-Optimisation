# Process-Optimisation

Script which optimises a planning of tasks over time, weighted according to priority, and contrained by an upper bound on resources that can be allocated in each time period. 

Assumptions:
  - Each task, say $A$, consists of sub-tasks which are indexed by the timeslot for which they are planned, say $a_2, a_3, a_4$. The sub-tasks must be performed in order, but may be shifted relative to their original planning
  - Each task has a priority score, or weight. This is a scalar greater than $1$.
  - A task must be performed in its entirety, or not at all
  - Each sub-task requires resources from (possibly multiple) resource pools
  - The objective function to be maximised, is resources utilised in the planning, weighted by task priority score (which is a parameter of the model, "weight_exponent")
  - The constraint is that, for each timeslot, we may not ask for more than some percentage of each resource pool's available resources (this percentage is a parameter called "ub" for upper bound)
