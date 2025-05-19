from dataclasses import dataclass

import pandas as pd

from ignite.engine import Engine
from ignite.metrics import Metric


@dataclass
class MetricAccumulator:
    metric_dict: dict[str, Metric]

    def __post_init__(self):
        self._metrics = {
            k: []
            for k in self.metric_dict.keys()
        }
        self._metrics["iteration"] = []
        self._metrics["epoch"] = []

    def __call__(self, engine: Engine):
        for name, metric in self.metric_dict.items():
            self._metrics[name].append(metric.compute())
        self._metrics["iteration"].append(engine.state.iteration)
        self._metrics["epoch"].append(engine.state.epoch)

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._metrics)
        return df
