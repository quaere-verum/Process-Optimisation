from utils.data_processing import (fetch_data,
                                   data_prep,
                                   remove_ongoing_projects,
                                   timeslot_options)
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
import pandas as pd
from typing import Union


def execute_optimisation(server: str, timeslot_capacity_multiplier: dict, capacity_upper_bound: float,
                         resource_ids: Union[str, int], weight_exponent: float, shift_bounds: tuple = (-np.inf, np.inf)):
    """
    Executes all sub-functions from the package to return the optimal planning.
    Parameters:
        - timeslot_capacity_multiplier: multiplies the capacity of each resource for a given timeslot by the given amount
        - capacity_upper_bound: multiplier for the maximum capacity requested from a resource (i.e. 1.3 = 130%)
        - weight_factor: determines how heavily priority is weighed
        - shift_bounds: determines by how much tasks may be shifted into the past/future to optimise the planning
        - resource: resource for which to optimise planning
    Returns:
        - selection: dataframe containing the selected new projects for the optimal planning. Multi-index: first layer
        is TaskID, second is timeslots shifted (negative means shifted into the past)
        - ongoing_tasks: tasks that have to be planned in because they are already running
        - all_tasks: all tasks
    """
    planning, available = fetch_data(resource_ids)
    all_tasks, resource_capacity = data_prep(planning, available)
    remaining_capacity, remaining_tasks, ongoing_tasks = remove_ongoing_projects(all_tasks,
                                                                                 resource_capacity,
                                                                                 timeslot_capacity_multiplier)
    # Clip capacity at 0, otherwise there will be no solution which does not exceed a resource's capacity bounds
    remaining_capacity = remaining_capacity.clip(lower=0, upper=np.inf)
    remaining_weights = remaining_tasks.groupby('TaskID').Weights.mean().to_frame()

    # Dataframe which lists, per project, how much time it asks from each resource, per timeslot
    timeslot_pivot = remaining_tasks.groupby(
        ['TaskID', 'Timeslot', 'ResourceID']).ResourceAmount.sum().unstack().unstack().fillna(value=0)
    selection = optimise_schedule(timeslot_pivot, remaining_weights, remaining_capacity,
                                  weight_exponent=weight_exponent, ub=capacity_upper_bound,
                                  shift_bounds=shift_bounds)
    return selection, ongoing_tasks, all_tasks


def optimise_schedule(data_pivot: pd.DataFrame, weights: pd.DataFrame, resource_capacity: pd.DataFrame,
                      weight_exponent: float, lb=0, ub=1.3, shift_bounds=(-np.inf, np.inf)):
    """
    This function solves the following problem: it selects the projects (contained in data_pivot) which maximise the
    amount of resources being utilised.
    The weight of a task makes the resources spent for this project weigh more heavily if they are planned in.
    The constraints are as follows:
        - The planning may not ask more from a resource than ub*resource_capacity
        - The tasks may not be shifted backwards more than shift_bounds[0],
        or shifted forwards more than shift_bounds[1]

    """
    # B is a dataframe whose rows are indexed by a tuple (TaskID, shift). The dataframe contains all tasks
    # and all its shifts within the given bounds.
    # Its columns are the resources and the timeslots, and its values are the resources demanded from a resource in
    # the given timeslot, for the given task (possibly shifted)
    # A2 is a matrix used for the constraints in the M.I.L.P. function. Currently, each task can be selected
    # multiple times, namely the amount of shifts that are possible.
    # A2 will ensure that only one of these options can be selected, i.e. no task can be planned
    # twice with different shifts.
    B, A2 = timeslot_options(data_pivot, shift_bounds)
    w = (weights.reindex(B.index.get_level_values(0)).values.reshape(-1)) ** weight_exponent
    resource_capacity_stacked = resource_capacity.fillna(value=0).stack().to_frame()
    C = resource_capacity_stacked.reindex(B.columns).fillna(value=0).values.reshape(-1)
    J = np.ones(B.shape[1])
    # A small amount (0.001) is added to the weights for the original planning.
    # That is, if it makes no difference for the constraints, then the original planning is preferred.
    c = (-B.dot(J)) * (w + np.where(np.array(B.index.get_level_values(1)) == 0, 0.001, 0).reshape(-1))
    A1 = B.T
    # Resource capacity may not be exceeded by more than ub
    constraint1 = LinearConstraint(A1, C * lb, C * ub)
    # Each project may only be planned once (i.e. not multiple shifted versions)
    constraint2 = LinearConstraint(A2, 0, 1)
    # The selection can only be 0 (not selected) or 1 (selected)
    bound = Bounds(lb=0, ub=1)
    problem = milp(c=c, bounds=bound, constraints=[constraint1, constraint2], integrality=1)
    return B.loc[pd.Series(problem.x.astype(bool), index=B.index)]
