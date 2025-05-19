from ignite.engine import Engine as IgniteEngine
from ignite.metrics import Metric


class Engine(IgniteEngine):
    metrics: dict[str, Metric]
