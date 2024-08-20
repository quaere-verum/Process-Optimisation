import pandas as pd
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from itertools import combinations
import warnings
from typing import Union, Literal, Tuple
from dataclasses import dataclass
from abc import abstractmethod
from tqdm import tqdm
warnings.filterwarnings('ignore', category=FutureWarning)
np.random.seed(123)
from src.base_schedule import ScheduleStatus
    
@dataclass
class ScheduleOptimiser(ScheduleStatus):
    """
    Simple schedule optimisation for items which have already been assigned to a timeslot. Goal is to select
    those items which maximise priority score while not exceeding resource constraints.
    Args:
        timeslots_available (int): available discrete timeslots over which to optimise the planning
        capacity_usage_bound (pd.DataFrame, pd.Series, float): upper bound on capacity usage per resource per timeslot (as multiplier of base)
        capacity_multiplier (pd.DataFrame): multiplier for resource capacity per timeslot
        all_items (pd.DataFrame): items to be planned in. Should contain the following columns:
            - 'Resource': which resource will execute the item
            - 'relativeCost': how much of the resource's capacity (across the entire time horizon) does this item cost
            - 'Priority': some numerical value indicating the priority of this item
            - 'Distribution': across which timeslots would this item be executed
    """
    def __post_init__(self):
        super().__post_init__()
        assert "Distribution" in self.all_items.columns, "Schedule optimiser assumes that items were already distributed over timeslots"

    def _options(self) -> pd.DataFrame:
        exploded = self.all_items.explode("Distribution")
        task_length = self.all_items["Distribution"].reindex(exploded.index)
        return exploded.pivot(columns=["Resource", "Distribution"], values="relativeCost").sort_index(axis=1).fillna(0) / task_length
    
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
    """
    Optimises a schedule for a collection of items that were not given an a priori distribution over available timeslots.
    Hence, this optimisation algorithm will execute the optimisation by trying to optimally distribute the items across
    available timeslots, while optimising priority score and not exceeding resource constraints. This grows the number of
    possible solutions exponentially, and therefore optimisation is done sequentially.
    Args:
        timeslots_available (int): available discrete timeslots over which to optimise the planning
        capacity_usage_bound (pd.DataFrame, pd.Series, float): upper bound on capacity usage per resource per timeslot (as multiplier of base)
        capacity_multiplier (pd.DataFrame): multiplier for resource capacity per timeslot
        all_items (pd.DataFrame): items to be planned in. Should contain the following columns:
            - 'Resource': which resource will execute the item
            - 'relativeCost': how much of the resource's capacity (across the entire time horizon) does this item cost
            - 'Priority': some numerical value indicating the priority of this item
        max_subplanning_size (int): how many items can be planned in at each step of the sequential optimisation (mostly to constrain CPU demand)
        timeslot_combinations (literal): whether it is possible to distribute an item across timeslots in every possible combinations (option 'all')
            or only those combinations where timeslots are adjacent (option 'adjacent')
        max_item_duration (int): maximum number of timeslots across which an item can be distributed
    """
    max_subplanning_size: int = 25
    timeslot_combinations: Literal["all", "adjacent"] = "all"
    max_item_duration: int = -1

    def __post_init__(self):
        super().__post_init__()
        if self.max_item_duration == -1:
            self.max_item_duration = self.timeslots_available
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
        self.reset()
        for step in tqdm(range(max_steps)):
            try:
                next(self)
            except StopIteration:
                print(f"Planning finished after {step} iterations.")
                break
        if display_on_finish:
            self.display_state()

@dataclass
class SubtaskOptimisation(ScheduleStatus):
    """
    Optimises a collection of subtasks, each of which belongs to some larger super-task. These super-tasks must be either entirely planned
    in, or entirely scrapped. The subtasks have an order in which they must be executed. The goal is to distribute the subtasks such that
    the priority score is optimised, the order is preserved, and resource constraints are not exceeded.
    Args:
        timeslots_available (int): available discrete timeslots over which to optimise the planning
        capacity_usage_bound (pd.DataFrame, pd.Series, float): upper bound on capacity usage per resource per timeslot (as multiplier of base)
        capacity_multiplier (pd.DataFrame): multiplier for resource capacity per timeslot
        all_items (pd.DataFrame): items to be planned in. Should contain the following columns:
            - 'Resource': which resource will execute the item
            - 'relativeCost': how much of the resource's capacity (across the entire time horizon) does this item cost
            - 'Priority': some numerical value indicating the priority of this item
            - 'TaskID': unique identifier for the super-task
            - 'Order': integer specifying which stage of the super-tasks this sub-task belongs to
    """
    def __post_init__(self):
        super().__post_init__()
        assert np.isin(["TaskID", "Order"], self.all_items.columns).all()
        self.main_task_dict = dict(zip(self.all_items["TaskID"].unique(), np.arange(len(self.all_items["TaskID"].unique()))))
        self.all_items["TaskID"] = self.all_items["TaskID"].map(self.main_task_dict)

    def _transform_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        agg_frame =  self.all_items.groupby("TaskID").agg(
            Priority=("Priority", np.mean), 
            relativeCost=("relativeCost", np.sum)
        )
        all_items_copy = self.all_items.copy()
        all_items_copy["Shifts"] = (self.timeslots_available - all_items_copy.groupby("TaskID").Order.transform("max")).apply(lambda x: np.arange(int(x)))
        exploded = all_items_copy.explode("Shifts")
        exploded["Timeslot"] = exploded["Order"] + exploded["Shifts"] + 1
        agg_cost = (exploded.groupby(["TaskID", "Shifts", "Timeslot", "Resource"])
            .relativeCost.sum().unstack().unstack()
            .sort_index(axis=0).sort_index(axis=1).fillna(0)
        )
        agg_score = -(agg_frame["Priority"] * agg_frame["relativeCost"]).reindex(agg_cost.index.get_level_values(0)).to_frame().sort_index()
        return agg_cost, agg_score

    def _make_selection(self):
        options, score = self._transform_data()
        selection_constraint = pd.concat(
           (options.index.get_level_values(0).to_series().reset_index(drop=True), 
            pd.Series(np.ones(len(options)))
            ), 
            axis=1
        ).pivot(columns="TaskID", values=0).T.fillna(0).astype(int).values
        planning = milp(
            score.values.reshape(-1),
            integrality=1,
            bounds=Bounds(0, 1),
            constraints=[
                LinearConstraint(selection_constraint, 0, 1),
                LinearConstraint(options.T.values, 0, self.remaining_capacity.unstack().values)
            ]
        )
        if planning.x is None:
            return 0
        final_selection = options.loc[pd.Series(np.round(planning.x).astype(bool), index=options.index)]
        self._update_state(final_selection)
        return len(final_selection)

    def _update_state(self, selected_items: pd.DataFrame):
        self.selection.extend(selected_items.index.to_list())
        current_selection = self.current_selection
        current_selection["Timeslot"] = current_selection["Order"] + current_selection["Shift"] + 1
        capacity_used = current_selection.groupby(["Timeslot", "Resource"]).relativeCost.sum().unstack()
        if self.capacity_used is None:
            self.capacity_used = capacity_used
        else:
            self.capacity_used = self.capacity_used + capacity_used
        self.remaining_capacity = (self.remaining_capacity - capacity_used).clip(0, np.inf)

    @property
    def current_selection(self) -> pd.DataFrame:
        current_selection = pd.DataFrame(self.selection, columns=['Key', 'Shift']).set_index('Key')
        current_selection = self.all_items.join(
            current_selection,
            on='TaskID',
            how='inner'
        )
        current_selection["Resource"] = current_selection["Resource"].map({v: k for k, v in self.resources_dict.items()})
        return current_selection

    def make_planning(self, display_on_finish: bool = False) -> None:
        self.reset()
        self._make_selection()
        if display_on_finish:
            self.display_state()