from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import utils
from pypfopt import risk_models
from rpy2.robjects import pandas2ri, r
from tqdm import tqdm

pandas2ri.activate()


r_code_dcc_garch = """
library(rmgarch)
library(dplyr)

forecast_dcc_garch_cov <- function(data_window, days_in_month) {
    # Define the GARCH specification
    spec <- ugarchspec(
      variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0))
    )

    # Create the multivariate GARCH specification
    num_columns <- ncol(data_window) - 1
    uspec <- multispec(replicate(num_columns, spec))
    dcc_spec <- dccspec(uspec, dccOrder = c(1, 1), distribution = 'mvnorm')

    # Fit the DCC-GARCH model
    dcc_fit <- dccfit(dcc_spec, data = data_window %>% select(-date))

    if (inherits(dcc_fit, 'uGARCHmultifit')) {
        # Handle non-convergence
        warning('DCC-GARCH fit did not converge. Returning NULL.')
        return(NULL)
    }

    # Forecast the DCC-GARCH model for days_in_month days
    n_ahead <- days_in_month
    dcc_forecast <- dccforecast(dcc_fit, n.ahead = n_ahead)
    dcc_cov_matrix <- rcov(dcc_forecast)[[1]]
    dcc_cov_matrix <- apply(dcc_cov_matrix, c(1, 2), sum)

    return(dcc_cov_matrix)
}
"""

r_code_go_garch = """
library(rmgarch)
library(dplyr)

forecast_go_garch_cov <- function(data_window, days_in_month) {
    spec <- ugarchspec(
        variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(0, 0))
    )

    # Create multispec for GO-GARCH
    num_columns <- ncol(data_window) - 1
    uspec <- multispec(replicate(num_columns, spec))
    # Specify the GO-GARCH model
    garch_spec <- gogarchspec(mean.model = 'constant',
                      variance.model = 'goGARCH',
                      distribution.model = 'mvnorm',
                      umodel = uspec)    
    # Fit the GO-GARCH model
    fit <- gogarchfit(spec = garch_spec, data = data_window %>% select(-date))

    if (inherits(fit, 'uGARCHmultifit')) {
        # Handle non-convergence
        warning('GO-GARCH fit did not converge. Returning NULL.')
        return(NULL)
    }

    # Forecast the GO-GARCH model
    n_ahead <- days_in_month
    gogarch_forecast <- gogarchforecast(fit, n.ahead = n_ahead)
    gogarch_cov_matrix <- rcov(gogarch_forecast)[[1]]
    gogarch_cov_matrix <- apply(gogarch_cov_matrix, c(1, 2), sum)

    return(gogarch_cov_matrix)
}
"""


# Execute the R code to define the functions in R environment
r(r_code_dcc_garch)
r(r_code_go_garch)


def forecast_dcc_garch_cov(data_window, days_in_month):
    # Convert the pandas DataFrame to R DataFrame
    r_data_window = pandas2ri.py2rpy(data_window.reset_index())
    dcc_cov_matrix = r["forecast_dcc_garch_cov"](r_data_window, days_in_month)
    if dcc_cov_matrix is None:
        return np.full(
            (data_window.shape[1], data_window.shape[1]), np.nan
        )  # Return NaN matrix if not converged
    return np.array(dcc_cov_matrix)


def forecast_go_garch_cov(data_window, days_in_month):
    # Convert the pandas DataFrame to R DataFrame
    r_data_window = pandas2ri.py2rpy(data_window.reset_index())
    go_garch_cov_matrix = r["forecast_go_garch_cov"](r_data_window, days_in_month)
    if go_garch_cov_matrix is None:
        return np.full(
            (data_window.shape[1], data_window.shape[1]), np.nan
        )  # Return NaN matrix if not converged
    return np.array(go_garch_cov_matrix)


def forecast_ra_cov(data_window, days_in_month):
    return risk_models.sample_cov(data_window, returns_data=True) / 252 * days_in_month


def forecast_shrinkage_cov(data_window, days_in_month):
    return (
        risk_models.CovarianceShrinkage(data_window, returns_data=True).ledoit_wolf()
        / 252
    ) * days_in_month


def forecast_ewma_cov(data_window, days_in_month, span=180):
    return (
        risk_models.exp_cov(data_window, returns_data=True, span=span) / 252
    ) * days_in_month


def realized_cov(df, date, days_in_month):
    cov_df = (
        df[(df.index.month == date.month) & (df.index.year == date.year)].astype(float)
        / 100
    )
    return (risk_models.sample_cov(cov_df, returns_data=True) / 252) * days_in_month


def process_date(df, d, L):
    print(f"Processing {d}")
    L_init = d + pd.DateOffset(months=-L)
    L_end = d + pd.DateOffset(days=-1)

    window = df.loc[L_init:L_end].astype(float) / 100
    days_in_month = d.days_in_month

    # Get forecasts
    ra_cov = forecast_ra_cov(window, days_in_month)
    shrinkage_cov = forecast_shrinkage_cov(window, days_in_month)
    ewma_cov = forecast_ewma_cov(window, days_in_month, span=180)
    dcc_garch_cov = forecast_dcc_garch_cov(window, days_in_month)
    go_garch_cov = forecast_go_garch_cov(window, days_in_month)

    # Get realized cov
    cov = realized_cov(df, d, days_in_month)

    # Calculate MSFE
    msfe_table = pd.DataFrame(
        data={
            "date": [d],
            "Rolling Average": [utils.calc_msfe(cov, ra_cov)],
            "LW Shrunk": [utils.calc_msfe(cov, shrinkage_cov)],
            "EWMA": [utils.calc_msfe(cov, ewma_cov)],
            "DCC-GARCH": [utils.calc_msfe(cov, dcc_garch_cov)],
            "GO-GARCH": [utils.calc_msfe(cov, go_garch_cov)],
        },
        index=[d],
    )
    return msfe_table


def forecast_cov_matrices(df, dates, L):
    msfe_list = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_date, df, d, L) for d in dates]

        for future in tqdm(as_completed(futures), total=len(dates)):
            msfe_list.append(future.result())

    return msfe_list
