# Author - Willis Knox
import requests
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

@dataclass
class BoxCox:
    apply: bool = False
    lam: Union[str, float] = 'auto'
    method: str = 'guerrero'
    upper: float = 2
    lower: float = -1

    def validate_inputs(self):
        if isinstance(self.lam, str):
            if self.lam.lower() != 'auto':
                self.lam = 'auto'
            self.upper = 2
            self.lower = -1
        if self.method.lower() != 'guerrero' or self.method.lower() != 'log_lik':
            self.method = 'guerrero'

    def to_json(self):
        json = self.__dict__
        json["lambda"] = json.pop("lam")
        return json

@dataclass
class Log:
    apply: bool = False
    base: Union[str, int] = 'e'
    constant: float = 1
    factor: Union[None, float] = None

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

@dataclass
class SeasonalDiff:
    apply: bool = False
    period: Union[str, int] = 'auto'
    n_diffs: Union[str, int] = 'auto'
    test: str = 'ss'
    alpha: float = 0.05

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


@dataclass
class FirstDiff:
    apply: bool = False
    n_diffs: Union[str, int] = 'auto'
    test: str = 'kpss'
    type: str = 'level'
    alpha: float = 0.05

    def validate_inputs(self):
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

@dataclass
class TimedoorTransformation:
    boxcox: BoxCox
    log: Log
    seasonal_diff: SeasonalDiff
    first_diff: FirstDiff

# global variables
BASE_URL: str = 'https://api.timedoor.io/invocation'
API_HEADER_KEY = 'X-Time-Door-Key'
HEADERS: dict = {}

def __run_request(url, json_data) -> Tuple[int, dict]:
    r = requests.post(url=url, json=json_data, headers=HEADERS)
    return (r.status_code, r.json())

def __convert_timeseries_to_data(dates: List[str], values: List[float]) -> Dict[str, float]:
    return dict(zip(dates, values))

def __clean_values(values, error_val: Union[int, float, None] = None) -> list:
    if not error_val:
        return values
    return [value if value != error_val else None for value in values]

def __validate_ma_window_size(size: int):
    if size < 2:
        return 2
    return size

def __set_api_key(key: Union[str, None]) -> bool:
    """Sets the api key for the user if it has not been set already

    Args:
        key (str): API key for the user
    """
    if key:
        HEADERS[API_HEADER_KEY] = key
        return True
    return False


def auto_arima(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
               error_value: Union[int, None] = None, reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant',
               stepwise: bool = True, approximation: bool = True, non_stationary: bool = True, seasonal: bool = True,
               ic: str = 'aicc', box_cox_lambda: Union[None, str, float] = None, bias_adj: bool = False,
               unit_root_test: str = 'kpss', seasonal_test: str = 'ss', alpha: float = 0.05,
               ci_level: int = 95, horizon: int = 10) -> Tuple[int, dict]:
    """ Forecasting

    univariate | nonparametric

    Automatic univariate time series forecasting with ARIMA models

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to None.
        imputation_window (int, optional): [description]. Defaults to None.
        error_value (Union[int, None], optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        stepwise (bool, optional): [description]. Defaults to True.
        approximation (bool, optional): [description]. Defaults to True.
        non_stationary (bool, optional): [description]. Defaults to True.
        seasonal (bool, optional): [description]. Defaults to True.
        ic (str, optional): [description]. Defaults to 'aicc'.
        box_cox_lambda (Union[None, str, float], optional): [description]. Defaults to None.
        bias_adj (bool, optional): [description]. Defaults to False.
        unit_root_test (str, optional): [description]. Defaults to 'kpss'.
        seasonal_test (str, optional): [description]. Defaults to 'ss'.
        alpha (float, optional): [description]. Defaults to 0.05.
        ci_level (int, optional): [description]. Defaults to 95.
        horizon (int, optional): [description]. Defaults to 10.

    Returns:
        Tuple[int, dict]: [description]
    """

    # check to see if API Key was provided
    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})
    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)

    url = f"{BASE_URL}/auto-arima"
    data = __convert_timeseries_to_data(dates=dates, values=values)

    # build json body
    body = {
        "stepwise": stepwise,
        "approximation": approximation,
        "seasonal": seasonal,
        "non_stationary": non_stationary,
        "ic": ic,
        "box_cox_lambda": box_cox_lambda,
        "bias_adj": bias_adj,
        "unit_root_test": unit_root_test,
        "seasonal_test": seasonal_test,
        "alpha": alpha,
        "ci_level": ci_level,
        "horizon": horizon,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)


def changepoint_detection(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear',
                          imputation_window: int = 10, error_value: Union[int, None] = None,
                          boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                          seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                          reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant',
                          penalty: str = 'mbic', min_distance: int = 1) -> Tuple[int, dict]:
    """Changepoint Detection

    univariate | nonparametric

    Detects multiple changepoints in the time series data using the changepoint detection algorithm Pruned Exact Linear Time (PELT)
    with a nonparametric cost function based on the empirical distribution of the data. PELT applies a penalty to prevent under- or over-fitting.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to None.
        imputation_window (int, optional): [description]. Defaults to None.
        error_value (Union[int, None], optional): [description]. Defaults to None.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        penalty (str, optional): [description]. Defaults to 'mbic'.
        min_distance (int, optional): [description]. Defaults to 1.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)

    url = f"{BASE_URL}/changepoint-detection"
    data = __convert_timeseries_to_data(dates=dates, values=values)

    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "penalty": penalty,
        "min_distance": min_distance,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)

def collective_and_point_anomalies(dates: List[str], values: List[float], api_key: Union[str, None] = None,
                                   imputation_method: str = 'linear', imputation_window: int = 10,
                                   error_value: Union[int, None] = None,
                                   boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                                   seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                                   reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant',
                                   method: str = 'mean_var', min_ca_size: int = 10) -> Tuple[int, dict]:
    """Anomaly Detection

    univariate | parametric

    Collective And Point Anomalies (CAPA) is an algorithm for detecting collective anomalies identified by a change in mean,
    variance, or both, and differentiating them from point anomalies.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'mean_var'.
        min_ca_size (int, optional): [description]. Defaults to 10.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)
    
    url = f"{BASE_URL}/collective-and-point-anomalies"
    data = __convert_timeseries_to_data(dates=dates, values=values)

    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "method": method,
        "min_ca_size": min_ca_size,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }
    
    return __run_request(url=url, json_data=body)


def conditional_heteroskedasticity(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear',
                                   imputation_window: int = 10, error_value: Union[int, None] = None,
                                   boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                                   seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                                   reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant',
                                   alpha: float = 0.05, window_size: int = 10) -> Tuple[int, dict]:
    """Early Warning Signal Detection

    univariate | nonparametric | fast data

    Estimates conditional heteroskedasticity within a moving window along the time series by fitting an autoregressive model using AIC optimization.
    Conditional heteroskedasticity is associated with the dynamical phenomena of rising variability and flickering.
    Conditional heteroskedasticity signifies variance with a positive relationship with variance at previous time steps.
    Given that variability tends to increase prior to a phase transition, conditional heteroskedasticity can serve as an early indicator
    since the part of a time series near an impending shift appears as a high variability cluster, while the part of the time series that moves away
    from the shift appears as a low variability cluster.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        alpha (float, optional): [description]. Defaults to 0.05.
        window_size (int, optional): [description]. Defaults to 10.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)
    
    url = f"{BASE_URL}/conditional-heteroskedasticity"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "alpha": alpha,
        "window_size": window_size,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)


def drift_diffusion_jump(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                         error_value: Union[int, None] = None,
                         boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                         seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                         reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant',
                         kernel_bandwidth_factor: float = 0.6, kernel_points: int = 500) -> Tuple[int, dict]:
    """Early Warning Signal Detection

    univariate | nonparametric

    The nonparametric Drift-Diffusion-Jump model models a wide variety of unknown underlying time series data generating processes and
    estimates drift, diffusion and jump metrics over data and time. These metrics can capture the dynamical phenomena of
    rising memory, rising variability, and flickering.

    The drift metric quantifies the instantaneous local rate of change in the time series.
    The diffusion metric quantifies the standard deviation of relatively small shocks at each time step.
    The jump metric quantifies relatively large intermittent shocks.
    Total variance combines diffusion and jumps. Conditional variance signifies variance with a positive relationship with variance
    at previous time steps and rises to infinity at a critical transition.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        kernel_bandwidth_factor (float, optional): [description]. Defaults to 0.6.
        kernel_points (int, optional): [description]. Defaults to 500.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)

    url = f"{BASE_URL}/drift-diffusion-jump"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "kernel_bandwidth_factor": kernel_bandwidth_factor,
        "kernel_points": kernel_points,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)


def early_warning_signals(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                          error_value: Union[int, None] = None,
                          boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                          seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                          reproduction: bool = False, precision_digits: int = 4,
                          precision_method: str = 'significant', method: str = 'acf1', window_size: int = 10) -> Tuple[int, dict]:
    """Early Warning Signal Detection

    univariate | parametric | fast data

    Estimates various statistical moments within a moving window along the time series.
    These estimates serve as potential early warning signals for critical transitions
    and are differently associated with the dynamical phenomena of rising memory, rising variability, and flickering.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'acf1'.
        window_size (int, optional): [description]. Defaults to 10.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)

    url = f"{BASE_URL}/early-warning-signals"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "method": method,
        "window_size": window_size,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                    "box_cox": boxcox_str,
                    "log": log_str,
                    "seasonal_diff": seasonal_diff_str,
                    "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)


def granger_causality(dates: List[str], x_values: List[float], y_values: List[float], api_key: Union[str, None] = None,
                      imputation_method: str = 'linear', imputation_window: int = 10,
                      error_value: Union[int, None] = None,
                      boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                      seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                      reproduction: bool = False, precision_digits: int = 4,
                      precision_method: str = 'significant', alpha: float = 0.05, gamma: float = 0.5) -> Tuple[int, dict]:
    """Causality Inference

    multivariate | parametric

    Computes variable-lag Granger causality to test if time series X Granger-causes time series Y.

    Granger-causality defines the causal relation between 2 time series in terms of predictability: X Granger-causes Y if X's past predicts Y's future better than Y's past alone.

    Variable-lag Granger causality allows X to influence Y with arbitrary instead of fixed time delays. Y is the first object (index = 0) in the ``time_series`` array, X the second object (index = 1).

    Args:
        dates (List[str]): [description]
        x_values (List[float]): [description]
        y_values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        alpha (float, optional): [description]. Defaults to 0.05.
        gamma (float, optional): [description]. Defaults to 0.5.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    x_values = __clean_values(values=x_values, error_val=error_value)
    y_values = __clean_values(values=y_values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)

    url = f"{BASE_URL}/granger-causality"
    x_data = __convert_timeseries_to_data(dates=dates, values=x_values)
    y_data = __convert_timeseries_to_data(dates=dates, values=y_values)

    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    body = {
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": y_data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                    "box_cox": boxcox_str,
                    "log": log_str,
                    "seasonal_diff": seasonal_diff_str,
                    "first_diff": first_diff_str
                }
            },
            {
                "data": x_data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                    "box_cox": boxcox_str,
                    "log": log_str,
                    "seasonal_diff": seasonal_diff_str,
                    "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)


def matrix_profile(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                   error_value: Union[int, None] = None,
                   boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                   seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                   reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant', method: str = 'stomp', window_size: int = 10,
                   exclusion_factor: float = 0.5, neighbor_exclusion_radius: int = 3, n_motifs: int = 3, n_motifs_neighbors: int = 10,
                   n_discords: int = 1, n_discord_neighbors: int = 3) -> Tuple[int, dict]:
    """Recurring Pattern Detection | Anomaly Detection | Chain Detection

    univariate | nonparametric | fast data

    Computes the Matrix Profile and searches for motifs, discords and chains.

    Relatively low ``distance`` values of the ``matrix_profile`` indicate that the subsequence in the original time series must have (at least one)
    relatively similar subsequence elsewhere in the data (such reoccurring patterns are called motifs).
    Relatively high ``distance`` values of the ``matrix_profile`` indicate that the subsequence in the original time series must be unique in its shape (such areas are called discords or anomalies). 

    Time series chains are motifs that evolve over time.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'stomp'.
        window_size (int, optional): [description]. Defaults to 10.
        exclusion_factor (float, optional): [description]. Defaults to 0.5.
        neighbor_exclusion_radius (int, optional): [description]. Defaults to 3.
        n_motifs (int, optional): [description]. Defaults to 3.
        n_motifs_neighbors (int, optional): [description]. Defaults to 3.

    Returns:
        Tuple[int, dict]: [description]
    """
    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)
    
    url = f"{BASE_URL}/matrix-profile"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "method": method,
        "window_size": window_size,
        "exclusion_factor": exclusion_factor,
        "neighbor_exclusion_radius": neighbor_exclusion_radius,
        "n_motifs": n_motifs,
        "n_motifs_neighbors": n_motifs_neighbors,
        "n_discords": n_discords,
        "n_discord_neighbors": n_discord_neighbors,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)

def serial_dependence(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                      error_value: Union[int, None] = None,
                      boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                      seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                      reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant', method: str = 'acf',
                      max_lag: Union[str, float] = '10*10log10(n)', alpha: float = 0.05) -> Tuple[int, dict]:
    """Serial Dependency Detection

    univariate | parametric

    Serial dependence provides functions for quantifying the strength of and evidence for linear and nonlinear lag-dependencies.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'acf'.
        max_lag (Union[str, float], optional): [description]. Defaults to '10*10log10(n)'.
        alpha (float, optional): [description]. Defaults to 0.05.

    Returns:
        Tuple[int, dict]: [description]
    """
    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)

    url = f"{BASE_URL}/serial-dependence"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "method": method,
        "max_lag": max_lag,
        "alpha": alpha,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)

def spectral_density(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                     error_value: Union[int, None] = None,
                     boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                     seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                     reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant', method: str = 'direct',
                     taper: str = 'rectangle', center: bool = True, conversion: Union[None, str] = None) -> Tuple[int, dict]:
    """Spectral Analysis

    univariate | parametric

    Estimates the spectral density of the time series via nonparametric models.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'direct'.
        taper (str, optional): [description]. Defaults to 'rectangle'.
        center (bool, optional): [description]. Defaults to True.
        conversion (Union[None, str], optional): [description]. Defaults to None.

    Returns:
        Tuple[int, dict]: [description]
    """
    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)
    
    url = f"{BASE_URL}/spectral-density"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "method": method,
        "taper": taper,
        "center": center,
        "conversion": conversion,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)

def spectral_entropy(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                     error_value: Union[int, None] = None,
                     boxcox: Union[BoxCox, None] = None, log: Union[Log, None] = None,
                     seasonal_diff: Union[SeasonalDiff, None] = None, first_diff: Union[FirstDiff, None] = None,
                     reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant', method: str = 'direct',
                     taper: str = 'rectangle', window_size: int = 10) -> Tuple[int, dict]:
    """Spectral Analysis

    univariate | parametric | fast data

    Estimates the spectral Shannon entropy from the normalized spectral density within a moving window along the time series.

    The ``Omega`` forecastability measure is defined as ``1 - spectral entropy``, thus an increase in spectral entropy corresponds to a decrease in forecastability.

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (BoxCox, optional): [description]. Defaults to None.
        log (Log, optional): [description]. Defaults to None.
        seasonal_diff (SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'direct'.
        taper (str, optional): [description]. Defaults to 'rectangle'.
        window_size (int, optional): [description]. Defaults to 10.

    Returns:
        Tuple[int, dict]: [description]
    """

    if not __set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = __clean_values(values=values, error_val=error_value)
    imputation_window = __validate_ma_window_size(imputation_window)
    
    url = f"{BASE_URL}/spectral-entropy"
    data = __convert_timeseries_to_data(dates=dates, values=values)
    
    boxcox_str = boxcox.to_json() if boxcox else None
    log_str = log.to_json() if log else None
    seasonal_diff_str = seasonal_diff.to_json() if seasonal_diff else None
    first_diff_str = first_diff.to_json() if first_diff else None

    # build json body
    body = {
        "method": method,
        "taper": taper,
        "window_size": window_size,
        "reproduction": reproduction,
        "precision": {
            "digits": precision_digits,
            "method": precision_method
        },
        "time_series": [
            {
                "data": data,
                "imputation": {
                    "method": imputation_method,
                    "ma_window_size": imputation_window
                },
                "transformations": {
                        "box_cox": boxcox_str,
                        "log": log_str,
                        "seasonal_diff": seasonal_diff_str,
                        "first_diff": first_diff_str
                }
            }
        ]
    }

    return __run_request(url=url, json_data=body)
