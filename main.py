from src.optimisation import SubtaskOptimisation
import numpy as np
import pandas as pd

length = 250
timeslots = 6
data = pd.DataFrame({
    "SubtaskID": [f"S{k}" for k in range(length)],
    "TaskID": [f"T{np.random.randint(low=0, high=length//5)}" for _ in range(length)],
    "Order": np.random.randint(low=0, high=timeslots, size=length),
    "relativeCost": np.random.uniform(low=2, high=10, size=length),
    "Resource": [f"Team {np.random.randint(low=0, high=5)}" for _ in range(length)],
    "Priority": np.random.randint(low=1, high=6, size=length)
}).set_index("SubtaskID")



capacity_multiplier = pd.DataFrame(columns=data["Resource"].unique(), index=np.arange(1, timeslots + 1), data=1)

if __name__ == "__main__":
    planner = SubtaskOptimisation(
        timeslots,
        capacity_multiplier=capacity_multiplier,
        capacity_usage_bound=1.1,
        all_items=data,

    )

    planner.make_planning(display_on_finish=True)

