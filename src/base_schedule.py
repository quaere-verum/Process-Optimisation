import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union
from dataclasses import dataclass
from abc import abstractmethod
import seaborn as sns
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
    """
    Base class for schedule optimisation, used to store the relevant variables and 
    implement basic and abstract methods common across all optimisation algorithms.
    """
    timeslots_available: int
    capacity_usage_bound: Union[pd.DataFrame, pd.Series, float]
    capacity_multiplier: pd.DataFrame
    all_items: pd.DataFrame

    
    def __post_init__(self):
        assert np.isin(["Resource", "relativeCost", "Priority"], self.all_items.columns).all()
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