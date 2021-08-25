from typing import Union

# TODO make all objects JSON Seralizeable
class BoxCox:
    def __init__(self, apply: bool = False, lam: Union[str, float] = 'auto',
                 method: str = 'guerrero', upper: float = 2, lower: float = -1) -> None:
        self.apply = apply
        self.lam = lam
        self.method = method.lower()
        self.upper = upper
        self.lower = lower
        self.validate_inputs()
        self.use_bounds = True
        if self.lam != 'auto':
            self.use_bounds = False

    def validate_inputs(self):
        if isinstance(self.lam, str):
            if self.lam.lower() != 'auto':
                self.lam = 'auto'
            self.upper = 2
            self.lower = -1
        if self.method.lower() != 'guerrero' or self.method.lower() != 'log_lik':
            self.method = 'guerrero'
    
    def to_json(self):
        return self.__dict__


class Log:
    def __init__(self, apply: bool = False, base: Union[str, int] = 'e',
                 factor: Union[None, float] = 0.0001, constant: float = 1) -> None:
        self.apply = apply
        self.base = base
        self.factor = factor
        self.constant = constant
        self.validate_inputs()

    def validate_inputs(self):
        if self.constant < 0.0001:
            self.constant = 0.0001
        if self.factor and self.factor < 0.0001:
            self.factor = 0.0001
        if isinstance(self.base, int):
            if self.base < 2:
                self.base = 2
            if self.base > 10:
                self.base = 10
        elif self.base != 'e':
            self.base = 'e'
    
    def to_json(self):
        return self.__dict__


class SeasonalDiff:
    def __init__(self, apply: bool = False, period: Union[str, int] = 'auto',
                 n_diffs: Union[str, int] = 'auto', test: str = 'ss', alpha: float = 0.05) -> None:
        self.apply = apply
        self.period = period
        self.n_diffs = n_diffs
        self.test = test.lower()
        self.alpha = alpha
        self.validate_inputs()

    def validate_inputs(self):
        if isinstance(self.period, str) and self.period != 'auto':
            self.period = 'auto'
        if isinstance(self.period, int) and self.period < 2:
            self.period = 2
        if isinstance(self.n_diffs, str) and self.n_diffs != 'auto':
            self.period = 'auto'
        if isinstance(self.n_diffs, int):
            if self.n_diffs < 1:
                self.n_diffs = 1
            elif self.n_diffs > 3:
                self.n_diffs = 3
        if self.test not in ['ss', 'ch', 'hegy', 'ocsb']:
            self.test = 'ss'
        if self.alpha < 0.01:
            self.alpha = 0.01
        elif self.alpha > 0.1:
            self.alpha = 0.1
    
    def to_json(self):
        return self.__dict__


class FirstDiff:
    def __init__(self, apply: bool = False, period: Union[str, int] = 'auto',
                 n_diffs: Union[str, int] = 'auto', test: str = 'kpss',
                 diff_type: str = 'level', alpha: float = 0.05) -> None:
        self.apply = apply
        self.period = period
        self.n_diffs = n_diffs
        self.test = test.lower()
        self.type = diff_type.lower()
        self.alpha = alpha
        self.validate_inputs()

    def validate_inputs(self):
        if isinstance(self.period, str) and self.period != 'auto':
            self.period = 'auto'
        if isinstance(self.period, int) and self.period < 2:
            self.period = 2
        if isinstance(self.n_diffs, str) and self.n_diffs != 'auto':
            self.period = 'auto'
        if isinstance(self.n_diffs, int):
            if self.n_diffs < 1:
                self.n_diffs = 1
            elif self.n_diffs > 3:
                self.n_diffs = 3
        if self.test not in ['kpss', 'adf', 'pp']:
            self.test = 'kpss'
        if self.type not in ['level', 'trend']:
            self.type = 'level'
        if self.alpha < 0.01:
            self.alpha = 0.01
        elif self.alpha > 0.1:
            self.alpha = 0.1
    
    def to_json(self):
        return self.__dict__


class TimedoorTransformation:
    def __init__(self, boxcox: BoxCox, log: Log, seasonal_diff: SeasonalDiff, first_diff: FirstDiff) -> None:
        self.boxcox = boxcox
        self.log = log
        self.seasonal_diff = seasonal_diff
        self.first_diff = first_diff


class TimedoorReproduction:
    pass


class TimedoorResponse:
    def __init__(self, computation_time, reproduction = None, data = None) -> None:
        self.computation_time: float = computation_time
        self.reproduction: TimedoorReproduction = reproduction
        self.data = data

    def __str__(self) -> str:
        return f'Response: {self.computation_time}, {self.reproduction}, {self.data}'


def main():
    pass


if __name__ == '__main__':
    main()
