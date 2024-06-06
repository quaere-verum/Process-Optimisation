import pandas as pd
import numpy as np
import os


def fetch_data(resource_ids):
    # Could add additional logic for retrieving data, e.g. querying the planned items for a start and end date
    # This was omitted from this demo file to preserve the abstraction
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
    if resource_ids is not None:
        planning = pd.read_csv(os.path.join(path, 'planning.csv')).query(f'ResourceID in {resource_ids}')
        available = pd.read_csv(os.path.join(path, 'available.csv')).query(f'ResourceID in {resource_ids}')
    else:
        planning = pd.read_csv(os.path.join(path, 'planning.csv'))
        available = pd.read_csv(os.path.join(path, 'available.csv'))
    return planning, available


def data_prep(planning: pd.DataFrame, available: pd.DataFrame):
    available = available[['Timeslot', 'ResourceID', 'ResourceAmount']]
    capacity = available.pivot_table(index=['ResourceID'], columns=['Timeslot'], values='ResourceAmount',
                                     aggfunc='sum', fill_value=0)
    planning = planning[['Timeslot', 'TaskID', 'TaskStatusID', 'ResourceID', 'ResourceAmount', 'Weights']]
    return planning, capacity


def remove_ongoing_projects(data: pd.DataFrame, resource_capacity: pd.DataFrame,
                            timeslot_capacity_multiplier: dict):

    # The TaskStatusID's which we have deemed "ongoing", so these have to be planned in
    ongoing_status = [3, 4, 5]

    # Change month_capacity_multiplier to a vector to be multiplier with resource_capacity

    timeslot_multiplier = resource_capacity.columns.to_series().map(timeslot_capacity_multiplier).sort_index().fillna(1)

    resource_timeslot_capacity = pd.DataFrame(resource_capacity.values*timeslot_multiplier.values,
                                              index=resource_capacity.index, columns=resource_capacity.columns)
    # Split tasks into ongoing and new, and adjust capacity accordingly
    ongoing_tasks = data.loc[data.TaskStatusID.isin(ongoing_status)].copy()
    # How much capacity is required for each time, per month, to finish the ongoing tasks
    ongoing_capacity_usage = ongoing_tasks.groupby(['ResourceID',
                                                    'Timeslot']).ResourceAmount.sum().unstack().fillna(value=0)
    ongoing_capacity_usage = ongoing_capacity_usage.reindex(columns=resource_timeslot_capacity.columns)
    # Adjust the capacity base on ongoing tasks
    remaining_capacity = resource_timeslot_capacity - ongoing_capacity_usage
    remaining_tasks = data.loc[~(data.TaskStatusID.isin(ongoing_status))].copy()
    return remaining_capacity, remaining_tasks, ongoing_tasks


def timeslot_options(data_pivot: pd.DataFrame, start_stop_times, bounds=(-np.inf, np.inf)):
    temp = {}
    max_timeslot = np.max(data_pivot.columns.get_level_values(1))
    # Calculate all the possible shifts for each task
    for k, task in enumerate(data_pivot.index):
        options = np.arange(-start_stop_times.loc[task, 'start'], max_timeslot - start_stop_times.loc[task, 'stop'] + 1)
        for s in options:
            temp[(task, int(s))] = data_pivot.loc[task].unstack().T.shift(int(s)).fillna(value=0)

    # Concatenate the resulting dataframes. Gives a dataframe whose rows
    # are indexed by a multi-index: TaskID and shift
    dataframe = pd.concat([pd.DataFrame(temp[key]).unstack() for key in temp], axis=1)
    dataframe.columns = pd.MultiIndex.from_tuples(list(temp.keys()))
    dataframe = dataframe.T
    arr = np.zeros(shape=(len(set(dataframe.index.get_level_values(0))), len(dataframe)))
    # Create an array which indicates how many times a TaskID occurs (after adding shifts).
    # Necessary for constraint in M.I.L.P. problem. Each problem may only be selected once
    curr_task = dataframe.index.get_level_values(0)[0]
    row = 0
    for col, i in enumerate(dataframe.index.get_level_values(0)):
        if i == curr_task:
            arr[row, col] = 1
        else:
            row += 1
            curr_task = i
            arr[row, col] = 1
    return dataframe, arr
