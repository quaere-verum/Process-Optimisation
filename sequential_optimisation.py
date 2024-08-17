import pandas as pd
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from itertools import combinations, product
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union, Dict, Tuple, Iterable, Callable, List
from dataclasses import dataclass
warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(123)



colour_dictionary = {
    'red' : '#ff0000',
    'orange' : '#ffa500',
    'white' : '#ffffff',
    'green' : '#63be7b',
    'light_blue' : '#add8e6',
    'dark_blue' : '#03b1fc'
}


@dataclass
class ScheduleStatus:
    timeslots_available: int
    capacity_usage_bound: Union[pd.DataFrame, pd.Series, float]
    capacity_multiplier: pd.DataFrame
    all_items: pd.DataFrame
    max_subplanning_size: int = 25
    
    def __post_init__(self):
        self.resources = list(self.all_items.Resource.unique())
        self.items_dict = dict(zip(list(self.all_items.index), np.arange(len(self.all_items))))
        self.resources_dict = dict(zip(self.resources, np.arange(len(self.resources))))
        self.all_items = self.all_items.rename(index=self.items_dict).sort_index()
        self.all_items['Resource'] = self.all_items['Resource'].map(self.resources_dict)

        self.remaining_items = self.all_items.copy()
        self.required_capacity: pd.DataFrame = None
        self.selection = []
        self.capacity_used: pd.DataFrame = None
        self.final_selection: pd.DataFrame = None

        self.required_capacity = pd.pivot(self.all_items, values=['relativeCost'], columns='Resource').fillna(0)
        self.required_capacity.columns = self.required_capacity.columns.get_level_values(1)
        self.required_capacity = self.required_capacity.sort_index(axis=0).sort_index(axis=1)
        self.item_score = self.all_items['relativeCost'] * self.all_items['Priority']
        self.item_score = self.item_score.sort_index()
        self.capacity_multiplier = self.capacity_multiplier.rename(columns=self.resources_dict)
        self.remaining_capacity = pd.DataFrame(
            100 / self.timeslots_available,
            columns=self.required_capacity.columns,
            index=np.arange(1, self.timeslots_available + 1)
        ) * self.capacity_multiplier * self.capacity_usage_bound

    def _update_state(self, selected_items: pd.DataFrame):
        capacity_used = selected_items.sum(axis=0).unstack().T.reindex(columns=self.remaining_capacity.columns, index=np.arange(1, self.timeslots_available + 1)).fillna(0)
        if self.capacity_used is None:
            self.capacity_used = capacity_used
        else:
            self.capacity_used = self.capacity_used + capacity_used
        self.remaining_capacity = (self.remaining_capacity - capacity_used).clip(0, np.inf)
        self.required_capacity = self.required_capacity.loc[~self.required_capacity.index.isin(selected_items.index.get_level_values(0))]
        self.item_score = self.item_score.loc[~self.item_score.index.isin(selected_items.index.get_level_values(0))]
        self.remaining_items = self.remaining_items.loc[~self.remaining_items.index.isin(selected_items.index.get_level_values(0))]
        self.selection.extend(selected_items.index.to_list())

    # TODO: Improve efficiency of this function
    def _generate_options(self, item_key: str, row: pd.Series) -> pd.DataFrame:
        combs = []
        for k in range(1, self.timeslots_available + 1):
            combs = combs + list(combinations(range(1, self.timeslots_available + 1), k))
        cols = pd.MultiIndex.from_tuples(list(product(np.arange(len(self.resources)), list(range(1, self.timeslots_available + 1)))))
        index = pd.MultiIndex.from_tuples(list(product([item_key], combs)))
        dataframe = pd.DataFrame(columns=cols, index=index)
        for ind, row2 in dataframe.iterrows():
            col = [(row['Resource'], k) for k in ind[1]]
            dataframe.loc[ind, col] = row['relativeCost'] / len(ind[1])
        return dataframe.fillna(0)

    # TODO: Improve efficiency of this function
    def _options(self, selection: pd.DataFrame) -> Union[pd.DataFrame, None]:
        options = [self._generate_options(ind, row) for ind, row in selection.iterrows()]
        if len(options) == 0:
            return None
        return pd.concat(options, axis=0).sort_index(axis=1).sort_index(axis=0)


class ScheduleOptimiser(ScheduleStatus):
    def solve_subplanning(self) -> Union[pd.Series, None]:
        constraint = self.remaining_capacity.sum(axis=0).values
        problem = milp(-self.item_score,
            integrality=1,
            bounds=Bounds(self.remaining_items["Mandatory"].astype(int), 1),
            constraints=[
                LinearConstraint(self.required_capacity.T, 0, constraint),
                LinearConstraint(np.ones(len(self.item_score)), 0, self.max_subplanning_size)
            ]
        )
        if problem.x is None:
            return None
        return pd.Series(np.round(problem.x).astype(bool), index=self.required_capacity.index)

    def selection_count_constraint(self, subplanning: pd.Series, options: pd.DataFrame) -> np.ndarray:
        selection_constraint = np.zeros((len(self.remaining_items.loc[subplanning]), len(options)))
        l = len(options) // len(self.remaining_items.loc[subplanning])
        for k in range(len(selection_constraint)):
            selection_constraint[k, k * l:(k + 1) * l] = 1
        return selection_constraint

    def make_selection(self) -> bool:
        subplanning = self.solve_subplanning()
        if subplanning is None:
            return False
        options = self._options(self.remaining_items.loc[subplanning])
        if options is None:
            return False
        selection_constraint = self.selection_count_constraint(subplanning, options)

        planning = milp(
            -self.item_score.reindex(options.index.get_level_values(0)).fillna(0),
            integrality=1,
            bounds=Bounds(0, 1),
            constraints=[
                LinearConstraint(selection_constraint, 0, 1),
                LinearConstraint(options.T.reindex(self.remaining_capacity.unstack().index).fillna(0), 0, self.remaining_capacity.unstack())
            ]
        )
        if planning.x is None:
            return False
        final_selection = options.loc[pd.Series(np.round(planning.x).astype(bool), index=options.index)]
        self._update_state(final_selection)
        
        return True

    def generate_planning(self, n_iter) -> None:
        for _ in range(n_iter):
            if len(self.remaining_items) == 0:
                break
            cont = self.make_selection()
            if not cont:
                break

        final_selection = pd.DataFrame(self.selection, columns=['Key', 'Distribution'])
        self.capacity_used.columns = self.capacity_used.columns.to_series().map({v: k for k, v in self.resources_dict.items()})
        self.final_selection = pd.merge(
            self.all_items.loc[self.all_items.index.to_series().isin(final_selection['Key'])], 
            final_selection.set_index('Key'),
            left_index=True, 
            right_index=True
        )
        inverse_dict = {v: k for k, v in self.items_dict.items()}
        self.final_selection.index = self.final_selection.index.to_series().map(inverse_dict)
                           
    def run(self, n_iter) -> None:
        self.generate_planning(n_iter)
        if self.final_selection is None:
            print(f'No feasible planning was found.')
        else:
            print(self.final_selection)
            print(f'Mean priority score original data: {np.mean(self.all_items["Priority"]):.3f}')
            print(f'Mean priority score planning: {np.mean(self.final_selection["Priority"]):.3f}')

        

if __name__ == '__main__':

    timeslots = 6
    capacity_bound = 1.1
    fake_item_keys = [f'Test-{k}' for k in range(1, 501)]
    fake_resources = [f'Team {k}' for k in range(1, 4)]
    fake_item_info = {key: {'relativeCost': np.random.randint(3, 10),
                                'Priority': np.random.randint(1, 101),
                                'Resource': np.random.choice(fake_teams, 1)[0],
                                'Mandatory': np.random.binomial(1, 0.01)}
                                for key in fake_feature_keys}
    capacity_multiplier = pd.DataFrame(1, columns=fake_resources, index=np.arange(1, timeslots + 1))

    planner = ScheduleOptimiser(
        timeslots_available=timeslots,
        capacity_usage_bound=capacity_bound,
        capacity_multiplier=capacity_multiplier,
        all_items=pd.DataFrame.from_dict(fake_item_info).T,
        max_subplanning_size=5,
    )
    planner.run(10)

