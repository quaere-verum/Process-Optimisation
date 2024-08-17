import pandas as pd
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from itertools import combinations, product
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union, Dict, Tuple, Iterable, Callable, List, Literal
from dataclasses import dataclass
from abc import abstractmethod
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors as mcolors
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
cmap = mcolors.ListedColormap([c for c in colour_dictionary.values()][::-1])


@dataclass
class ScheduleStatus:
    timeslots_available: int
    capacity_usage_bound: Union[pd.DataFrame, pd.Series, float]
    capacity_multiplier: pd.DataFrame
    all_items: pd.DataFrame

    
    def __post_init__(self):
        if self.max_item_duration == -1:
            self.max_item_duration = self.timeslots_available
        self.resources = list(self.all_items.Resource.unique())
        self.items_dict = dict(zip(list(self.all_items.index), np.arange(len(self.all_items))))
        self.resources_dict = dict(zip(self.resources, np.arange(len(self.resources))))
        self.all_items = self.all_items.rename(index=self.items_dict).sort_index()
        self.all_items['Resource'] = self.all_items['Resource'].map(self.resources_dict)
        self.required_capacity = pd.pivot(self.all_items, values=['relativeCost'], columns='Resource').fillna(0)
        self.required_capacity.columns = self.required_capacity.columns.get_level_values(1)
        self.required_capacity = self.required_capacity.sort_index(axis=0).sort_index(axis=1)
        self.item_score = self.all_items['relativeCost'] * self.all_items['Priority']
        self.item_score = self.item_score.sort_index()
        self.capacity_multiplier = self.capacity_multiplier.rename(columns=self.resources_dict)

        self.reset()
        
        
    def reset(self):
        self.remaining_items = self.all_items.copy()
        self.selection = []
        self.capacity_used: pd.DataFrame = None
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
    
    def display_state(self):
        capacity_used = self.capacity_used.rename(columns={v: k for k, v in self.resources_dict.items()})
        print(self.current_selection)
        sns.heatmap(capacity_used, vmin=60 / self.timeslots_available, vmax=140 / self.timeslots_available, annot=True, cmap=cmap)
        plt.show()

    @property
    def current_selection(self) -> pd.DataFrame:
        current_selection = pd.DataFrame(self.selection, columns=['Key', 'Distribution'])
        current_selection = pd.merge(
            self.all_items.loc[current_selection['Key'].to_list()], 
            current_selection.set_index('Key'),
            left_index=True, 
            right_index=True
        ).rename(index={v: k for k, v in self.items_dict.items()})
        current_selection["Resource"] = current_selection["Resource"].map({v: k for k, v in self.resources_dict.items()})
        return current_selection
    
    @abstractmethod
    def make_planning(self) -> None:
        raise NotImplementedError
    

class ScheduleOptimiser(ScheduleStatus):
    def _options(self) -> pd.DataFrame:
        assert "Distribution" in self.all_items.columns, "Schedule optimiser assumes that items were already distributed over timeslots"
        return self.all_items.explode("Distribution").pivot(columns=["Resource", "Distribution"], values="relativeCost").sort_index(axis=1).fillna(0)
    
    def make_planning(self, display_on_finish: bool = False):
        if self.capacity_used is not None:
            self.reset()
        options = self._options()
        planning = milp(
            -self.item_score.fillna(0),
            integrality=1,
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(
                options.T.reindex(self.remaining_capacity.unstack().index).fillna(0), 
                0, 
                self.remaining_capacity.unstack()
            )
        )
        selected_items = options.loc[pd.Series(np.round(planning.x).astype(bool), index=options.index)]
        selected_items.index = pd.MultiIndex.from_arrays([selected_items.index, self.all_items.loc[selected_items.index].Distribution.values])
        self._update_state(selected_items)
        if display_on_finish:
            self.display_state()

@dataclass
class SequentialScheduleOptimiser(ScheduleStatus):
    max_subplanning_size: int = 25
    timeslot_combinations: Literal["all", "adjacent"] = "all"
    max_item_duration: int = -1

    def __post_init__(self):
        super().__post_init__()
        if self.timeslot_combinations == "all":
            combs = []
            for k in range(1, self.max_item_duration + 1):
                combs = combs + list(combinations(range(1, self.timeslots_available + 1), k))
        elif self.timeslot_combinations == "adjacent":
            combs = [tuple(np.arange(1, length + 1) + offset) for length in range(1, self.max_item_duration + 1) 
                     for offset in range(self.timeslots_available) 
                     if offset + length <= self.timeslots_available]
            
        index = self.all_items.index.to_list() * len(combs)
        item_timeslot_distributions = self.all_items.reindex(index).sort_index().set_index(pd.MultiIndex.from_product([self.all_items.index, combs]))
        item_timeslot_distributions["Distribution"] = item_timeslot_distributions.index.get_level_values(1).values
        all_options = item_timeslot_distributions.explode("Distribution").pivot(columns=["Resource", "Distribution"], values="relativeCost").sort_index(axis=1)
        all_options = all_options.div(all_options.index.get_level_values(1).to_series().apply(lambda x: len(x)).values.reshape(-1, 1))
        self._all_options = all_options

    def _solve_subplanning(self) -> Union[pd.Series, None]:
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
        return self.remaining_items.loc[pd.Series(np.round(problem.x).astype(bool), index=self.remaining_items.index)].index

    def _selection_count_constraint(self, options: pd.DataFrame) -> np.ndarray:
       return pd.concat(
           (options.index.get_level_values(0).to_series().reset_index(drop=True), 
            pd.Series(np.ones(len(options)))
            ), 
            axis=1
        ).pivot(columns=0, values=1).T.fillna(0).astype(int).values

    def _make_selection(self) -> int:
        subplanning = self._solve_subplanning()
        if subplanning is None:
            return 0
        options = self._all_options.loc[subplanning]
        selection_constraint = self._selection_count_constraint(options)

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
            return 0
        final_selection = options.loc[pd.Series(np.round(planning.x).astype(bool), index=options.index)]
        self._update_state(final_selection)
        
        return len(final_selection)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.remaining_items) == 0:
            raise StopIteration
        nr_selected = self._make_selection()
        if nr_selected == 0:
            raise StopIteration
        
    def make_planning(self, max_steps: int = 10, display_on_finish: bool = False) -> None:
        for step in tqdm(range(max_steps)):
            try:
                next(self)
            except StopIteration:
                print(f"Planning finished after {step} iterations.")
                break
        if display_on_finish:
            self.display_state()

        

if __name__ == '__main__':

    timeslots = 6
    capacity_bound = 1.1
    fake_item_keys = [f'Test-{k}' for k in range(1, 501)]
    fake_resources = [f'Team {k}' for k in range(1, 7)]
    fake_item_info = {key: {'relativeCost': np.random.randint(3, 10),
                                'Priority': np.random.randint(1, 101),
                                'Resource': np.random.choice(fake_resources, 1)[0],
                                'Mandatory': np.random.binomial(1, 0.01)}
                                for key in fake_item_keys}
    capacity_multiplier = pd.DataFrame(1, columns=fake_resources, index=np.arange(1, timeslots + 1))

    planner = SequentialScheduleOptimiser(
        timeslots_available=timeslots,
        capacity_usage_bound=capacity_bound,
        capacity_multiplier=capacity_multiplier,
        all_items=pd.DataFrame.from_dict(fake_item_info).T,
        max_subplanning_size=20,
        max_item_duration=4,
        timeslot_combinations="adjacent",
    )
    planner.make_planning(display_on_finish=True)
