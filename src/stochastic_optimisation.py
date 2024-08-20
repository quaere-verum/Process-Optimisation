from base_schedule import ScheduleStatus
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Literal
import scipy.stats as ss
from scipy.optimize import milp, Bounds, LinearConstraint


@dataclass
class StochasticScheduleStatus(ScheduleStatus):
    scenario_capacity_bounds: pd.DataFrame
    buckets: pd.DataFrame
    distribution: Literal["normal", "uniform", "log_normal", "exponential"]
    confidence_quantile: float = 0.95


    def __post_init__(self):
        super().__post_init__()
        assert np.isin(["Resource", "Bucket", "Priority", "TimeslotDistribution"], self.all_items.columns).all()
        assert np.isin(["QuantileLow", "QuantileHigh"], self.scenario_capacity_bounds.columns).all()
        assert np.isin([resource for resource in self.resources_dict.keys()], self.scenario_capacity_bounds.index).all()
        
        self._infer_distributions(distribution=self.distribution)

        self.item_score = self.all_items['Expectation'] * self.all_items['Priority']
        self.item_score = self.item_score.sort_index()


    def _infer_distributions(self, distribution: Literal["normal", "uniform", "log_normal", "exponential"]):
        match distribution:
            case "normal":
                assert np.isin(["mu", "sigma"], self.buckets.columns).all()
                self.all_items["Expectation"] = self.buckets["mu"].loc[self.all_items["Bucket"].values]
                self.all_items["StandardDeviation"] = self.buckets["sigma"].loc[self.all_items["Bucket"].values]
                q_lower = (1 - self.confidence_quantile) / 2
                q_upper = (1 + self.confidence_quantile) / 2
                self.all_items["QuantileLow"] = ss.norm.cdf(
                    q_lower, 
                    loc=self.all_items["Expectation"].values, 
                    scale=self.all_items["StandardDeviation"].values
                )
                self.all_items["QuantileHigh"] = ss.norm.cdf(
                    q_upper, 
                    loc=self.all_items["Expectation"].values, 
                    scale=self.all_items["StandardDeviation"].values
                )
            case "uniform":
                assert np.isin(["a", "b"], self.buckets.columns).all()
                raise NotImplementedError
            case "exponential":
                assert np.isin("lambda", self.buckets.columns).all()
                raise NotImplementedError
            case "log_normal":
                assert np.isin(["mu", "sigma"], self.buckets.columns).all()
                raise NotImplementedError
            
        
class UncorrelatedStochasticOptimiser(StochasticScheduleStatus):
    def _options(self) -> pd.DataFrame:
        exploded = self.all_items.explode("TimeslotDistribution")
        task_length = self.all_items["TimeslotDistribution"].reindex(exploded.index)
        return exploded.pivot(
            columns=["Resource", "TimeslotDistribution"], 
            values=["Expectation", "StandardDeviation", "QuantileLow", "QuantileHigh"]
        ).sort_index(axis=1).fillna(0) / task_length
    
    def make_planning(self, display_on_finish: bool = False):
        if self.capacity_used is not None:
            self.reset()
        options = self._options()
        planning = milp(
            -self.item_score.fillna(0),
            integrality=1,
            bounds=Bounds(0, 1),
            constraints=[
                LinearConstraint(
                    options["QuantileLow"].T.reindex(self.remaining_capacity.unstack().index).fillna(0), 
                    self.scenario_capacity_bounds["QuantileLow"], 
                    self.remaining_capacity.unstack()
                ),
                LinearConstraint(
                    options["QuantileHigh"].T.reindex(self.remaining_capacity.unstack().index).fillna(0), 
                    0, 
                    self.scenario_capacity_bounds["QuantileHigh"] 
                ),
                LinearConstraint(
                    options["Expectation"].T.reindex(self.remaining_capacity.unstack().index).fillna(0), 
                    0, 
                    self.remaining_capacity.unstack() 
                )
            ]
        )
        selected_items = options.loc[pd.Series(np.round(planning.x).astype(bool), index=options.index)]
        selected_items.index = pd.MultiIndex.from_arrays([selected_items.index, self.all_items.loc[selected_items.index].Distribution.values])
        self._update_state(selected_items)
        if display_on_finish:
            self.display_state()