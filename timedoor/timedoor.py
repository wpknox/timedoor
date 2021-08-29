# Author - Willis Knox
import requests
import timedoor.timedoor_objects as td
from typing import Dict, List, Tuple, Union

# TODO: replace method responses with proper object?

# global variables
BASE_URL: str = 'https://api.timedoor.io'
API_HEADER_KEY = 'X-Time-Door-Key'
HEADERS: dict = {}


def run_request(url, json_data) -> Tuple[int, dict]:
    r = requests.post(url=url, json=json_data, headers=HEADERS)
    return (r.status_code, r.json())


def convert_timeseries_to_data(dates: List[str], values: List[float]) -> Dict[str, float]:
    return dict(zip(dates, values))


def clean_values(values, error_val: Union[int ,float] = None) -> list:
    if not error_val: return values
    return [value if value != error_val else None for value in values]


def validate_ma_window_size(size: int):
    if size < 2:
        return 2
    return size

def set_api_key(key: str) -> bool:
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
    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})
    values = clean_values(values=values, error_val=error_value)
    imputation_window = validate_ma_window_size(imputation_window)

    url = BASE_URL+'/invocation/auto-arima'
    data = convert_timeseries_to_data(dates=dates, values=values)
    
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

    r = run_request(url=url, json_data=body)
    return r


def changepoint_detection(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                          error_value: Union[int, None] = None, transformation: td.TimedoorTransformation = None, reproduction: bool = False,
                          precision_digits: int = 4, precision_method: str = 'significant', penalty: str = 'mbic', min_distance: int = 1) -> Tuple[int, dict]:

    return changepoint_detection(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                                 boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                                 first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                                 precision_method=precision_method, penalty=penalty, min_distance=min_distance)


def changepoint_detection(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                          error_value: Union[int, None] = None, boxcox: td.BoxCox = td.BoxCox(), log: td.Log = td.Log(),
                          seasonal_diff: td.SeasonalDiff = td.SeasonalDiff(), first_diff: td.FirstDiff = td.FirstDiff(),
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

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)
    imputation_window = validate_ma_window_size(imputation_window)

    url = BASE_URL+'/invocation/changepoint-detection'
    data = convert_timeseries_to_data(dates=dates, values=values)

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
                "transformations":
                    {
                        "box_cox": boxcox.to_json(),
                        "log": log.to_json(),
                        "seasonal_diff": seasonal_diff.to_json(),
                        "first_diff": first_diff.to_json()
                }
            }
        ]
    }

    r = run_request(url=url, json_data=body)
    return r


def collective_and_point_anomalies(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                                   error_value: Union[int, None] = None,
                                   transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                                   precision_method: str = 'significant', method: str = 'mean_var', min_ca_size: int = 10) -> Tuple[int, dict]:

    return collective_and_point_anomalies(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                                          boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                                          first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                                          precision_method=precision_method, method=method, min_ca_size=min_ca_size)


def collective_and_point_anomalies(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                                   error_value: Union[int, None] = None,
                                   boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})
    
    values = clean_values(values=values, error_val=error_value)
    imputation_window = validate_ma_window_size(imputation_window)

    pass


def conditional_heteroskedasticity(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                                   error_value: Union[int, None] = None,
                                   transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                                   precision_method: str = 'significant', alpha: float = 0.05, window_size: int = 10) -> Tuple[int, dict]:

    return conditional_heteroskedasticity(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                                          boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                                          first_diff=transformation.first_diff, reproduction=reproduction,
                                          precision_digits=precision_digits, precision_method=precision_method, alpha=alpha, window_size=window_size)


def conditional_heteroskedasticity(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                                   error_value: Union[int, None] = None,
                                   boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        alpha (float, optional): [description]. Defaults to 0.05.
        window_size (int, optional): [description]. Defaults to 10.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)
    imputation_window = validate_ma_window_size(imputation_window)

    pass


def drift_diffusion_jump(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                         error_value: Union[int, None] = None,
                         transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                         precision_method: str = 'significant', kernel_bandwidth_factor: float = 0.6, kernel_points: int = 500) -> Tuple[int, dict]:

    return drift_diffusion_jump(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                                boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff, first_diff=transformation.first_diff,
                                reproduction=reproduction, precision_digits=precision_digits, precision_method=precision_method,
                                kernel_bandwidth_factor=kernel_bandwidth_factor, kernel_points=kernel_points)


def drift_diffusion_jump(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                         error_value: Union[int, None] = None,
                         boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        kernel_bandwidth_factor (float, optional): [description]. Defaults to 0.6.
        kernel_points (int, optional): [description]. Defaults to 500.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)

    pass


def early_warning_signals(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                          error_value: Union[int, None] = None,
                          transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                          precision_method: str = 'significant', method: str = 'acf1', window_size: int = 10) -> Tuple[int, dict]:

    return early_warning_signals(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                                 boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                                 first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                                 precision_method=precision_method, method=method, window_size=window_size)


def early_warning_signals(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                          error_value: Union[int, None] = None,
                          boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'acf1'.
        window_size (int, optional): [description]. Defaults to 10.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)

    pass


def granger_causality(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                      error_value: Union[int, None] = None,
                      transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                      precision_method: str = 'significant', alpha: float = 0.05, gamma: float = 0.5) -> Tuple[int, dict]:

    return granger_causality(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                             boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                             first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                             precision_method=precision_method, alpha=alpha, gamma=gamma)


def granger_causality(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                      error_value: Union[int, None] = None,
                      boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
                      reproduction: bool = False, precision_digits: int = 4,
                      precision_method: str = 'significant', alpha: float = 0.05, gamma: float = 0.5) -> Tuple[int, dict]:
    """Causality Inference

    multivariate | parametric

    Computes variable-lag Granger causality to test if time series X Granger-causes time series Y.

    Granger-causality defines the causal relation between 2 time series in terms of predictability: X Granger-causes Y if X's past predicts Y's future better than Y's past alone.

    Variable-lag Granger causality allows X to influence Y with arbitrary instead of fixed time delays. Y is the first object (index = 0) in the ``time_series`` array, X the second object (index = 1).

    Args:
        dates (List[str]): [description]
        values (List[float]): [description]
        api_key (Union[str, None], optional): [description]. Defaults to None.
        imputation_method (str, optional): [description]. Defaults to 'linear'.
        imputation_window (int, optional): [description]. Defaults to 10.
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        alpha (float, optional): [description]. Defaults to 0.05.
        gamma (float, optional): [description]. Defaults to 0.5.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)

    pass


def matrix_profile(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                   error_value: Union[int, None] = None,
                   transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                   precision_method: str = 'significant', method: str = 'stomp', window_size: int = 10, exclusion_factor: float = 0.5,
                   neighbor_exclusion_radius: int = 3, n_motifs: int = 3, n_motifs_neighbors: int = 3) -> Tuple[int, dict]:

    return matrix_profile(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                          boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                          first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                          precision_method=precision_method, method=method, window_size=window_size, exclusion_factor=exclusion_factor,
                          neighbor_exclusion_radius=neighbor_exclusion_radius, n_motifs=n_motifs, n_motifs_neighbors=n_motifs_neighbors)


def matrix_profile(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                   error_value: Union[int, None] = None,
                   boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
                   reproduction: bool = False, precision_digits: int = 4, precision_method: str = 'significant', method: str = 'stomp', window_size: int = 10,
                   exclusion_factor: float = 0.5, neighbor_exclusion_radius: int = 3, n_motifs: int = 3, n_motifs_neighbors: int = 3) -> Tuple[int, dict]:
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
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
        Union[td.TimedoorResponse, str]: [description]
    """
    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)
    pass


def serial_dependence(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                      error_value: Union[int, None] = None,
                      transformation: td.TimedoorTransformation = None, reproduction: bool = False,
                      precision_digits: int = 4, precision_method: str = 'significant', method: str = 'acf',
                      max_lag: Union[str, float] = '10*10log10(n)', alpha: float = 0.05) -> Tuple[int, dict]:

    return serial_dependence(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                             boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                             first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                             precision_method=precision_method, method=method, max_lag=max_lag, alpha=alpha)


def serial_dependence(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                      error_value: Union[int, None] = None,
                      boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'acf'.
        max_lag (Union[str, float], optional): [description]. Defaults to '10*10log10(n)'.
        alpha (float, optional): [description]. Defaults to 0.05.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """
    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)

    pass


def spectral_density(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                     error_value: Union[int, None] = None,
                     transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                     precision_method: str = 'significant', method: str = 'direct', taper: str = 'rectangle',
                     center: bool = True, conversion: Union[None, str] = None) -> Tuple[int, dict]:

    return spectral_density(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                            boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                            first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                            precision_method=precision_method, method=method, taper=taper, center=center, conversion=conversion)


def spectral_density(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                     error_value: Union[int, None] = None,
                     boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'direct'.
        taper (str, optional): [description]. Defaults to 'rectangle'.
        center (bool, optional): [description]. Defaults to True.
        conversion (Union[None, str], optional): [description]. Defaults to None.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """
    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)
    pass


def spectral_entropy(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                     error_value: Union[int, None] = None,
                     transformation: td.TimedoorTransformation = None, reproduction: bool = False, precision_digits: int = 4,
                     precision_method: str = 'significant', method: str = 'direct', taper: str = 'rectangle', window_size: int = 10) -> Tuple[int, dict]:

    return spectral_entropy(dates=dates, values=values, api_key=api_key, imputation_method=imputation_method, imputation_window=imputation_window, error_value=error_value,
                            boxcox=transformation.boxcox, log=transformation.log, seasonal_diff=transformation.seasonal_diff,
                            first_diff=transformation.first_diff, reproduction=reproduction, precision_digits=precision_digits,
                            precision_method=precision_method, method=method, taper=taper, window_size=window_size)


def spectral_entropy(dates: List[str], values: List[float], api_key: Union[str, None] = None, imputation_method: str = 'linear', imputation_window: int = 10,
                     error_value: Union[int, None] = None,
                     boxcox: td.BoxCox = None, log: td.Log = None, seasonal_diff: td.SeasonalDiff = None, first_diff: td.FirstDiff = None,
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
        boxcox (td.BoxCox, optional): [description]. Defaults to None.
        log (td.Log, optional): [description]. Defaults to None.
        seasonal_diff (td.SeasonalDiff, optional): [description]. Defaults to None.
        first_diff (td.FirstDiff, optional): [description]. Defaults to None.
        reproduction (bool, optional): [description]. Defaults to False.
        precision_digits (int, optional): [description]. Defaults to 4.
        precision_method (str, optional): [description]. Defaults to 'significant'.
        method (str, optional): [description]. Defaults to 'direct'.
        taper (str, optional): [description]. Defaults to 'rectangle'.
        window_size (int, optional): [description]. Defaults to 10.

    Returns:
        Union[td.TimedoorResponse, str]: [description]
    """

    if not set_api_key(api_key):
        return (400, {"message": 'api key was not provided!'})

    values = clean_values(values=values, error_val=error_value)

    imputation_window = validate_ma_window_size(imputation_window)
    pass


def main():
    pass


if __name__ == '__main__':
    main()
