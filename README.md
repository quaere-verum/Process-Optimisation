# Process-Optimisation

Script which optimises a planning of tasks over time, weighted according to priority, and contrained by an upper bound on resources that can be allocated in each time period. 

Assumptions:
  - Each task, say $A$, consists of sub-tasks which are indexed by the timeslot for which they are planned, say $a_2, a_3, a_4$
  - Sub-tasks must be performed in order, but may be shifted relative to their original planning
  - Each task has a priority score, or weight. This is a scalar greater than $1$
  - A task must be performed in its entirety, or not at all
  - Each task has a TaskStatusID. If a TaskStatusID has been labeled "ongoing", this task must be scheduled in its entirety
  - Each sub-task requires resources from (possibly multiple) resource pools
  - The objective function to be maximised, is resources utilised in the planning, weighted by task priority score (which is a parameter of the model, "weight_exponent")
  - The constraint is that, for each timeslot, we may not ask for more than some percentage of each resource pool's available resources (this percentage is a parameter called "ub" for upper bound)

Example: projects that need to be worked on by multiple different teams, divided up into sub-tasks. Resources are FTE.

This is the base version which only handles one (or, if you carefully check the code, two) constraint(s); it assumes each resource pool has one type of resource (e.g. FTE in the example) to spend. One can easily modify the code to include additional constraints, e.g. budgetary constraints in the example above.
