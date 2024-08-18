# Process-Optimisation

Contains several classes, each of which performs a specific optimisation routine. In each case, we face the following scenario:

- We have some collection of items $\{a_i\}_{0 \leq i\leq N}$ that we want to plan in
- Each item costs some number of resources to be executed
- Each item has some priorty ranking
- We have a finite time horizon over which to execute some sub-selection of the items
- We have finite resources to constrain the number of items that we can execute


The repository contains classes which solve this problem through mixed integer linear programming. We have three flavours.
### 1. Pre-assigned timeslots (ScheduleOptimiser)
In the first case, we assume that all the tasks have already been assigned to some timeslots (possible multiple per task), and our task is simply to choose the subset which optimises the priority score.

### 2. No pre-assigned timeslots (SequentialScheduleOptimiser)
The assumption above is abandoned. Doing so increases the number of possibilities drastically, because we now also have to choose how to distribute the items across timeslots. This increase is linear in the number of items but $O(2^T)$ in the number of timeslots $T$, since $2^T$ is the cardinality of the power set of $\{1,\dots,T\}$. Therefore, this class cuts the problem into multiple sub-problems which may as a result exclude the global optimum, but can still find a good solution to the problem.

### 3. Sub-tasks and super-tasks (SubtaskOptimisation)
Here the assumption is that each item is a sub-task that is part of some larger super-task. The goal is to schedule the super-tasks by distributing them across timeslots, but we are now constrained by preserving the ordering of the sub-tasks (as well as the resources and finite time). The super-tasks have to be scheduled in their entirety, or not at all. They can be shifted within the bounds of the finite time horizon, but the ordering of the sub-tasks cannot be altered.