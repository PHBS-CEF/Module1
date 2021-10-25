#!/usr/bin/env python
# coding: utf-8

import datetime as dt
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:
    print("Warning! Numba not found, routines will not be speed-enhanced")
    def njit(f):
        return f


@njit
def next_state(s_t, alpha, beta):
    """
    Transitions from employment/unemployment in period t to
    employment/unemployment in period t+1
    
    Parameters
    ----------
    s_t : int
        The individual's current state... s_t = 0 maps to
        unemployed and s_t = 1 maps to employed
    alpha : float
        The probability that an individual goes from
        unemployed to employed
    beta : float
        The probability that an individual goes from
        employed to unemployed

    Returns
    -------
    s_tp1 : int
        The individual's employment state in `t+1`
    """
    # Draw a random number
    u_t = random.random()

    # Let 0 be unemployed... If unemployed and draws
    # a value less than lambda then becomes employed
    if (s_t == 0) and (u_t < alpha):
        return 1
    # Let 1 be employed... If employed and draws a
    # value less than beta then becomes unemployed
    elif (s_t == 1) and (u_t < beta):
        return 0
    # Otherwise, he keeps the same state as he had
    # at period t
    else:
        return s_t


@njit
def simulate_employment_history(alpha, beta, s_0):
    """
    Simulates the history of employment/unemployment. It
    will simulate as many periods as elements in `alpha`
    and `beta`
    
    Parameters
    ----------
    alpha : np.array(float, ndim=1)
        The probability that an individual goes from
        unemployed to employed
    beta : np.array(float, ndim=1)
        The probability that an individual goes from
        employed to unemployed
    s_0 : int
        The initial state of unemployment/employment, which
        should take value of 0 (unemployed) or 1 (employed)
    """
    # Create array to hold the values of our simulation
    assert(len(alpha) == len(beta))
    T = len(alpha)
    s_hist = np.zeros((T+1,), np.int8)

    s_hist[0] = s_0
    for t in range(T):
        # Step one period into the future
        s_0 = next_state(s_0, alpha[t], beta[t])  # Notice alpha[t] and beta[t]
        s_hist[t+1] = s_0

    return s_hist


@njit
def count_frequencies_individual(history):
    """
    Computes the transition probabilities for a two-state
    Markov chain

    Parameters
    ----------
    history : np.array(int, ndim=1)
        An array with the state values of a two-state Markov chain
    
    Returns
    -------
    alpha : float
        The probability of transitioning from state 0 to 1
    beta : float
        The probability of transitioning from state 1 to 0
    """
    # Get length of the simulation and an index tracker
    T = len(history)
    idx = np.arange(T)

    # Determine when the chain had values 0 and 1 -- Notice
    # that we can't use the last value because we don't see
    # where it transitions to
    zero_idxs = idx[(history == 0) & (idx < T-1)]
    one_idxs = idx[(history == 1) & (idx < T-1)]

    # Check what percent of the t+1 values were 0/1
    alpha = np.sum(history[zero_idxs+1]) / len(zero_idxs)
    beta = np.sum(1 - history[one_idxs+1]) / len(one_idxs)

    return alpha, beta

@njit
def check_accuracy(T, alpha=0.25, beta=0.025):
    """
    Checks the accuracy of our fit by printing the true values
    and the fitted values for a given T
    
    Parameters
    ----------
    T : int
        The length of our simulation
    alpha : float
        The probability that an individual goes from
        unemployed to employed
    beta : float
        The probability that an individual goes from
        employed to unemployed
    """
    idx = np.arange(T)
    alpha_np = np.ones(T)*alpha
    beta_np = np.ones(T)*beta

    # Simulate a sample history
    emp_history = simulate_employment_history(alpha_np, beta_np, 0)

    # Check the fit
    alpha_hat, beta_hat = count_frequencies_individual(emp_history)
    
    print(f"True alpha was {alpha} and fitted value was {alpha_hat}")
    print(f"True beta was {beta} and fitted value was {beta_hat}")
    
    return alpha, alpha_hat, beta, beta_hat


@njit
def simulate_employment_cross_section(alpha, beta, s_0, N=500):
    """
    Simulates a cross-section of employment/unemployment using
    the model we've described above.
    
    Parameters
    ----------
    alpha : np.array(float, ndim=1)
        The probability that an individual goes from
        unemployed to employed
    beta : np.array(float, ndim=1)
        The probability that an individual goes from
        employed to unemployed
    s_0 : np.array(int, ndim=1)
        The fraction of the population that begins in each
        employment state
    N : int
        The number of individuals in our cross-section
    
    Returns
    -------
    s_hist_cs : np.array(int, ndim=2)
        An `N x T` matrix that contains an individual
        history of employment along each row
    """
    # Make sure transitions are same size and get the length
    # of the simulation from the length of the transition
    # probabilities
    assert(len(alpha) == len(beta))
    T = len(alpha)

    # Check the fractions add to one and figure out how many
    # zeros we should have
    assert(np.abs(np.sum(s_0) - 1.0) < 1e-8)
    Nz = int(math.floor(s_0[0]*N))

    # Allocate space to store the simulations
    s_hist_cs = np.zeros((N, T+1), np.int8)
    s_hist_cs[Nz:, 0] = 1
    
    for i in range(N):
        s_hist_cs[i, :] = simulate_employment_history(
            alpha, beta, s_hist_cs[i, 0]
        )
    
    return s_hist_cs


def pandas_employment_cross_section(eu_ue_df, s_0, N=500):
    """
    Simulate a cross-section of employment experiences
    
    Parameters
    ----------
    eu_ue_df : pd.DataFrame
        A DataFrame with columns `dt`, `alpha`, and `beta`
        that have the monthly eu/ue transition rates
    s_0 : np.array(float, ndim=1)
        The fraction of the population that begins in each
        employment state
    N : int
        The numbers of individuals in our cross-section

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with the dates and an employment outcome
        associated with each date of `eu_ue_df`
    """
    # Make sure that `ue_ue_df` is sorted by date
    eu_ue_df = eu_ue_df.sort_values("dt")
    alpha = eu_ue_df["alpha"].to_numpy()
    beta = eu_ue_df["beta"].to_numpy()

    # Simulate cross-section
    employment_history = simulate_employment_cross_section(
        alpha, beta, s_0, N
    )

    df = pd.DataFrame(employment_history[:, :-1].T)
    df = pd.concat([eu_ue_df["dt"], df], axis=1)
    df = pd.melt(
        df, id_vars=["dt"],
        var_name="pid", value_name="employment"
    )

    return df



def cps_interviews(df, start_year, start_month):
    """
    Takes an individual simulated employment/unemployment
    history and "interviews" the individual as if they were
    in the CPS
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with at least the columns `pid`, `dt`,
        and `employment`
    start_year : int
        The year in which their interviewing begins
    start_month : int
        The month in which their interviewing begins

    Returns
    -------
    cps : pd.DataFrame
        A DataFrame with the same columns as `df` but only
        with observations that correspond to the CPS
        interview schedule for someone who starts
        interviewing in f`{start_year}/{start_month}`
    """
    # Get dates that are associated with being interviewed in
    # the CPS
    start_date_y1 = dt.datetime(start_year, start_month, 1)
    dates_y1 = pd.date_range(start_date_y1, periods=4, freq="MS")
    start_date_y2 = dt.datetime(start_year+1, start_month, 1)
    dates_y2 = pd.date_range(start_date_y2, periods=4, freq="MS")
    dates = dates_y1.append(dates_y2)

    # Filter data that's not in the dates
    cps = df.loc[df["dt"].isin(dates), :]

    return cps

def cps_count_frequencies(df):
    """
    Estimates the transition probability from employment
    and unemployment histories of a CPS sample of
    individuals
    
    Parameters
    ----------
    df : pd.DataFrame
        A sample of individuals from the CPS survey. Must
        have columns `dt`, `pid`, and `employment`.

    Returns
    -------
    alpha : float
        The probability of transitioning from unemployment
        to employment
    beta : float
        The probability of transitioning from employment
        to unemployment
    """
    # Set the index to be dt/pid
    data_t = df.set_index(["dt", "pid"])

    # Now find the "t+1" months and "pid"s
    tp1 = data_t.index.get_level_values("dt").shift(periods=1, freq="MS")
    pid = data_t.index.get_level_values("pid")
    idx = pd.MultiIndex.from_arrays([tp1, pid], names=["dt", "pid"])

    # Now "index" into the data and reset index
    data_tp1 = (
        data_t.reindex(idx)
            .rename(columns={"employment": "employment_tp1"})
    )
    out = pd.concat(
        [
            data_t.reset_index().loc[:, ["dt", "pid", "employment"]],
            data_tp1.reset_index()["employment_tp1"]
        ], axis=1, sort=True
    ).dropna(subset=["employment_tp1"])
    out["employment_tp1"] = out["employment_tp1"].astype(int)

    # Count how frequently we go from 0 to 1
    out_zeros = out.query("employment == 0")
    alpha = out_zeros["employment_tp1"].mean()
    
    # Count how frequently we go from 1 to 0
    out_ones = out.query("employment == 1")
    beta = (1 - out_ones["employment_tp1"]).mean()

    return alpha, beta


def check_accuracy_cs(N, T, alpha=0.25, beta=0.025):
    """
    Checks the accuracy of our fit by printing the true values
    and the fitted values for a given T
    
    Parameters
    ----------
    N : int
        The total number of people we ever interview
    T : int
        The length of our simulation
    alpha : float
        The probability that an individual goes from
        unemployed to employed
    beta : float
        The probability that an individual goes from
        employed to unemployed
    """
    alpha_beta_df = pd.DataFrame(
        {
            "dt": pd.date_range("2018-01-01", periods=T, freq="MS"), 
            "alpha": np.ones(T)*alpha,
            "beta": np.ones(T)*beta
        }
    )

    # Simulate the full cross-section
    frac_unemployed = beta / (alpha + beta)
    frac_employed = alpha / (alpha + beta)
    df = pandas_employment_cross_section(
        alpha_beta_df, np.array([frac_unemployed, frac_employed]), N
    )

    # Interview individuals according to the cps interviews
    interview = lambda x: cps_interviews(
        x,
        np.random.choice(df["dt"].dt.year.unique()),
        np.random.randint(1, 13)
    )
    cps_data = (
        df.groupby("pid")
          .apply(
              lambda x: interview(x)
          )
          .reset_index(drop=True)
    )

    # Check the fit
    alpha_hat, beta_hat = cps_count_frequencies(cps_data)
    
    print(f"True alpha was {alpha} and fitted value was {alpha_hat}")
    print(f"True beta was {beta} and fitted value was {beta_hat}")
    
    return alpha, alpha_hat, beta, beta_hat
