# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 2025

@author: Yu Eva Zhang
"""

# Standard library imports
import os
import glob
import pickle
import random
import warnings
from os.path import dirname, join as pjoin

# Data processing and numerical computation
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.io import loadmat
from scipy.stats import zscore, sem
from scipy.optimize import minimize, basinhopping

# Machine learning imports
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

# Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Visualization setup
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

# Configure warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning) # I may not always want to do this


# prepare history matrix
def prepare_hist_matrix(a, R, n_back):
    num_trials = len(a)
    # Convert actions to -1/1 coding and mask non-choice trials
    a_coded = np.zeros(num_trials)
    a_coded[a==1] = 1
    a_coded[a==2] = -1
    
    # Create history matrices
    a_hist = np.zeros((num_trials, n_back))
    R_hist = np.zeros((num_trials, n_back))
    uR_hist = np.zeros((num_trials, n_back))
    
    # Fill matrices
    for n in range(n_back):
        a_hist[(1+n):, n] = a_coded[:-1-n]
        R_hist[(1+n):, n] = R[:-1-n]
        uR_hist[(1+n):, n] = (R==0)[:-1-n] & (a_coded!=0)[:-1-n]
    
    return np.fliplr(a_hist), np.fliplr(R_hist), np.fliplr(uR_hist)


# Basic logistic Regression fitting functions
def logistic_prepare_predictors(a, R, n_back):
    # Get history matrices
    a_hist, R_hist, _ = prepare_hist_matrix(a, R, n_back)
    
    # Create predictors
    choice_mask = (a==1) | (a==2)
    past_r = np.multiply(a_hist, R_hist)
    past_ur = np.multiply(a_hist, R_hist==0)
    
    # Create target (-1 for right, 1 for left)
    target = np.zeros(len(a))
    target[a==1] = 1
    target[a==2] = -1
    
    # Filter choice trials and combine predictors
    glm_mat = np.concatenate((past_r[choice_mask,:], past_ur[choice_mask,:]), axis=1)
    glm_target = target[choice_mask]
    
    return glm_mat, glm_target

def win_lose(a, R):
    a_cleaned = a.copy()
    a_cleaned[(a==3) | (a==4)] = 0  # Set non-choice trials to zero
    
    # Get next trial's choices and rewards
    post_choice = np.append(a_cleaned[1:], 0)
    
    # Calculate basic conditions
    choice_trials = (a_cleaned != 0)
    next_choice_trials = (post_choice != 0)
    win = (R == 1) & choice_trials & next_choice_trials
    lose = (R == 0) & choice_trials & next_choice_trials
    
    # Calculate stay/switch
    stay = (post_choice == a_cleaned) & choice_trials & next_choice_trials
    switch = (post_choice != a_cleaned) & choice_trials & next_choice_trials
    
    # Calculate combined conditions
    win_stay = stay & win
    lose_switch = switch & lose
    win_switch = switch & win
    lose_stay = stay & lose
    
    # Calculate rates
    win_stay_rate = np.sum(win_stay) / np.sum(win)
    lose_stay_rate = np.sum(lose_stay) / np.sum(lose)
    lose_switch_rate = np.sum(lose_switch) / np.sum(lose)
    win_switch_rate = np.sum(win_switch) / np.sum(win)
    
    # Calculate normalized metrics
    win_stay_nm = win_stay_rate * 2 / (win_stay_rate + lose_stay_rate + 1e-10)
    lose_switch_nm = lose_switch_rate * 2 / (lose_switch_rate + win_switch_rate + 1e-10)
    
    return win_stay_nm, lose_switch_nm, win_stay_rate, lose_stay_rate, lose_switch_rate, win_switch_rate

def prediction_accuracy_RL(a, R, PP):
    choice_mask = (a==1) | (a==2)
    return np.mean((a[choice_mask]==1) == (PP[choice_mask,0] > 0.5))

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def runLength(a,R):
    choice_trials = (a==1)|(a==2)
    a_choice=a[choice_trials]
    R_choice=R[choice_trials]
    
    reward_rate = np.count_nonzero(R_choice == 1)/len(R_choice)

    b = np.diff(a_choice)
    condition = (b == 0) & (a_choice[:-1]!=3) & (a_choice[:-1]!=4)
    runL = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2] + 1
    run_range = zero_runs(b)
    
    num_rew_list = []
    num_unr_list = []
    frac_rew_list = []
    frac_unr_list = []

    for i in range(len(run_range)):
        R_run = R_choice[run_range[i][0]:run_range[i][1]+1]
        num_rew = np.count_nonzero(R_run == 1)
        num_unr = np.count_nonzero(R_run == 0)
        num_rew_list.append(num_rew)
        num_unr_list.append(num_unr)
        frac = num_rew/(num_rew+num_unr)
        frac_rew_list.append(frac)
        frac_unr_list.append(1-frac)
    
    return runL, num_rew_list, num_unr_list, frac_rew_list, frac_unr_list, reward_rate

def load_sessions(list_of_files):
    # Initialize regression coefficients DataFrame
    coef_columns = ([f'RewC{i}' for i in range(10, 0, -1)] + 
                   [f'UnrC{i}' for i in range(10, 0, -1)])
    coef_df = pd.DataFrame(columns=coef_columns)
    
    # Initialize storage lists
    alpha_rew_all, alpha_unr_all, decay_all = [], [], []
    win_stay_nm_all, lose_switch_nm_all = [], []
    win_stay_all, lose_stay_all = [], []
    win_switch_all, lose_switch_all = [], []
    reward_rate_list = []
    prediction_acc_RL_list, prediction_acc_LR_list = [], []
    intercept_all = []
    miss_rate_list, alarm_rate_list = [], []
    mouse_ids = []  # New list for mouse IDs
    total_choice_trials_list = []  # New list for total choice trials
    brain_areas = []  # New list for brain areas
    reaction_times_list = []  # New list for average reaction times
    total_rewarded_trials_list = []  # New list for total rewarded trials
    
    # RL model setup
    initial_params = np.array([np.log(0.2), np.log(0.2), 1, 0, np.log(0.2)])
    bounds = [(-np.inf,0), (-np.inf,0), (0, 15), (-1, 1), (-np.inf,0)]
    
    # Process each session
    for ss in range(len(list_of_files)):
        # Extract mouse ID from filename (first 5 characters, e.g., 'RH055')
        mouse_id = list_of_files[ss][:5]
        mouse_ids.append(mouse_id)
        
        # Extract brain area from filename (4th position when split by underscore)
        filename_parts = os.path.basename(list_of_files[ss]).split('_')
        brain_area = filename_parts[3] if len(filename_parts) > 3 else 'Unknown'
        brain_areas.append(brain_area)
        
        # Load session data
        data_dict = loadmat(list_of_files[ss])
        R, a = data_dict['R'][0], data_dict['a'][0]
        
        # Extract reaction time data
        session_data = data_dict['SessionData']
        
        # Use length of actions array for n_trials
        n_trials = len(a)
        reaction_time = np.zeros(n_trials)
        ready_duration = np.zeros(n_trials)
        
        # Access trial events directly
        trial_events = session_data['RawEvents'][0][0]['Trial']
        
        # Loop through each trial to extract reaction time
        for tt in range(n_trials):
            # Check if trial_events has enough trials
            if tt >= len(trial_events[0][0][0]):
                reaction_time[tt] = np.nan
                continue
            
            # Get current trial
            current_trial = trial_events[0][0][0][tt]
            events = current_trial['Events'][0][0]
            states = current_trial['States'][0][0]
            
            # Extract left and right lick times
            left_lick = np.inf
            right_lick = np.inf
            
            # Check if Port1Out (left lick) exists
            if 'Port1Out' in events.dtype.names and events['Port1Out'].size > 0:
                left_lick = events['Port1Out'][0][0][0][0]
            
            # Check if Port2Out (right lick) exists
            if 'Port2Out' in events.dtype.names and events['Port2Out'].size > 0:
                right_lick = events['Port2Out'][0][0][0][0]
            
            # If neither lick exists, set reaction time to NaN
            if left_lick == np.inf and right_lick == np.inf:
                reaction_time[tt] = np.nan
                continue
            
            # Calculate Ready state duration
            ready_duration[tt] = states['Ready'][0][0][0][1] - states['Ready'][0][0][0][0]
            reaction_time[tt] = min(left_lick, right_lick) - ready_duration[tt]

        # Calculate average reaction time for the session (excluding NaN values)
        # Only include reaction times from choice trials (a==1 or a==2)
        choice_trials = (a==1) | (a==2)
        choice_reaction_times = reaction_time[choice_trials]
        
        # Filter out negative reaction times (due to occassional bug in bpod software that might happen 1-2 times per session)
        valid_reaction_times = choice_reaction_times[(~np.isnan(choice_reaction_times)) & (choice_reaction_times >= 0)]
        
        # Calculate median from valid (non-negative) reaction times
        median_reaction_time = np.median(valid_reaction_times) if len(valid_reaction_times) > 0 else np.nan
        reaction_times_list.append(median_reaction_time)
        
        # Count total choice trials (a==1 or a==2)
        choice_trials = (a==1) | (a==2)
        total_choice_trials = np.sum(choice_trials)
        total_choice_trials_list.append(total_choice_trials)
        
        # Calculate miss and alarm rates
        miss_rate = np.mean(a == 4)  # Calculate miss rate
        alarm_rate = np.mean(a == 3)  # Calculate alarm rate
        miss_rate_list.append(miss_rate)
        alarm_rate_list.append(alarm_rate)
        
        # Fit RL model and get parameters
        lik_model = minimize(RW_full, initial_params, args=(a, R, 1), method='Powell', bounds=bounds)
        nloglik, QQ, PP = RW_full(lik_model.x, a, R, 0)
        
        # Store RL parameters
        alpha_rew_all.append(np.exp(lik_model.x[0]))
        alpha_unr_all.append(np.exp(lik_model.x[1]))
        decay_all.append(np.exp(lik_model.x[4]))
        
        # Calculate and store prediction accuracy
        prediction_acc_RL_list.append(prediction_accuracy_RL(a, R, PP))
        
        # Calculate and store win-stay/lose-switch metrics
        win_lose_metrics = win_lose(a, R)
        win_stay_nm_all.append(win_lose_metrics[0])
        lose_switch_nm_all.append(win_lose_metrics[1])
        win_stay_all.append(win_lose_metrics[2])
        lose_stay_all.append(win_lose_metrics[3])
        win_switch_all.append(win_lose_metrics[4])
        lose_switch_all.append(win_lose_metrics[5])
        
        # Calculate run length metrics and reward rate
        _, _, _, _, _, reward_rate = runLength(a, R)
        reward_rate_list.append(reward_rate)
        
        # Calculate total rewarded trials
        total_rewarded_trials = reward_rate * total_choice_trials
        total_rewarded_trials_list.append(total_rewarded_trials)
        
        # Prepare data for logistic regression
        glm_mat, glm_target = logistic_prepare_predictors(a, R, 10)
        
        # Current option: No regularization
        log_reg = LogisticRegression(solver='lbfgs', penalty='none', 
                                   n_jobs=-1, multi_class='auto', 
                                   fit_intercept=True, max_iter=10000)
        
        # Fit once and store results
        log_reg.fit(glm_mat, glm_target)
        coef_df.loc[ss,:] = log_reg.coef_[0]
        intercept_all.append(log_reg.intercept_[0])
        
        # Calculate prediction accuracy using sklearn metrics
        predictions = log_reg.predict(glm_mat)
        prediction_acc_LR_list.append(metrics.accuracy_score(glm_target, predictions))
    
    return (
        alpha_rew_all, alpha_unr_all, decay_all, 
        win_stay_nm_all, lose_switch_nm_all, coef_df,
        intercept_all, win_stay_all, lose_stay_all, 
        win_switch_all, lose_switch_all, reward_rate_list,
        prediction_acc_RL_list, prediction_acc_LR_list,
        miss_rate_list, alarm_rate_list, mouse_ids,
        total_choice_trials_list, brain_areas, reaction_times_list,
        total_rewarded_trials_list  # Added total_rewarded_trials_list
    )

# Plotting functions
def _setup_axis_style(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=12)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add reference line
    ax.axhline(y=0, color='k', linestyle=':')

def plot_raw_weights(common_x, coef_df, intercept):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, 
                            sharex=False, tight_layout=True)
    
    n_sessions = len(coef_df)
    
    for ss in range(n_sessions):
        # Calculate color based on session index
        color = (ss/n_sessions, 0, (n_sessions-ss)/n_sessions)
        
        # Plot RewC weights
        axes[0].plot(common_x, coef_df.iloc[ss, 0:10].values,
                    color=color, alpha=0.6, lw=2)
        
        # Plot UnrC weights
        axes[1].plot(common_x, coef_df.iloc[ss, 10:20].values,
                    color=color, alpha=0.6, lw=2)
        
        # Plot bias
        axes[2].scatter(0, intercept[ss], color=color, alpha=0.6, lw=1)
    
    # Set up styling for each axis
    _setup_axis_style(axes[0], "Past Trials", r"$\beta_{RewC(t-i)}$", "RewC")
    _setup_axis_style(axes[1], "Past Trials", r"$\beta_{UnrC(t-i)}$", "UnrC")
    _setup_axis_style(axes[2], "Left Bias", r"$\beta_{0}$", "Bias")
    
    return fig, axes


def box_plot(data, edge_color, fill_color):
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, patch_artist=True)
    
    # Set colors for all elements
    plt.setp(bp.values(), color=edge_color)
    
    # Set fill color for boxes
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color, alpha=0.7)
    
    return bp

def plot_comparison(df1, df2, label1, label2, save_path=None):  
    # Setup
    common_x = np.arange(-10, 0, 1)
    colors = {
        'group1': '#2ca02c',  # Medium-dark green
        'group2': '#FFC107'   # Saturated golden yellow
    }
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(5, 3.2), 
                            sharey=True, sharex=True, tight_layout=True)
    
    # Helper function to plot group data
    def plot_group_data(df, color, label):
        for i, ax in enumerate(axes):
            start_idx = i * 10
            end_idx = start_idx + 10
            
            means = df.mean(axis=0)[start_idx:end_idx]
            sems = [sem(df.iloc[:, w+start_idx]) for w in range(10)]
            
            ax.errorbar(common_x, means, yerr=sems,
                       c=color, marker='o', markersize=3,
                       linestyle='-', lw=2, capsize=4, label=label)
    
    # Plot both groups
    plot_group_data(df1, colors['group1'], label1)
    plot_group_data(df2, colors['group2'], label2)
    
    # Set up axes
    _setup_axis_style(axes[0], "Past Trials", r"$\beta_{RewC(t-i)}$", "RewC(t-i)")
    _setup_axis_style(axes[1], "Past Trials", r"$\beta_{UnrC(t-i)}$", "UnrC(t-i)")
    
    # Set x-ticks and y-limits
    axes[0].set_xticks([-10, -5, -1])
    axes[1].set_xticks([-10, -5, -1])
    axes[0].set_ylim([-1.5, 4])
    axes[1].set_ylim([-1.5, 4])
    
    # Save figures if save_path is provided
    if save_path is not None:
        fig.savefig(save_path + '.png')
        fig.savefig(save_path + '.svg')
    
    return fig, axes

def analyze_coefficient_differences(df1, df2, label1, label2, mouse_ids1=None, mouse_ids2=None, alpha=0.05):
    import statsmodels.formula.api as smf
    import pandas as pd
    import numpy as np
    
    # Check if column names match
    if not all(df1.columns == df2.columns):
        raise ValueError("Column names in df1 and df2 must match")
    
    # Print summary of mouse data
    print(f"Group 1 ({label1}): {len(df1)} sessions, {len(set(mouse_ids1))} unique mice")
    print(f"Group 2 ({label2}): {len(df2)} sessions, {len(set(mouse_ids2))} unique mice")
    
    # Store p-values and significance markers
    p_values = {}
    significance = {}
    
    # Process each coefficient column
    for col in df1.columns:
        # Create combined DataFrame for mixed-effects model
        combined_data = []
        
        # Add data from group 1
        for i, val in enumerate(df1[col]):
            combined_data.append({
                'mouse_id': mouse_ids1[i],
                'group': label1,
                'value': val
            })
        
        # Add data from group 2
        for i, val in enumerate(df2[col]):
            combined_data.append({
                'mouse_id': mouse_ids2[i],
                'group': label2,
                'value': val
            })
        
        # Convert to DataFrame
        model_df = pd.DataFrame(combined_data)
        
        try:
            # Print diagnostic information
            print(f"Analyzing {col}: {len(model_df)} observations across {model_df['mouse_id'].nunique()} mice")
            
            # Simple mixed-effects model with random intercept for mouse_id
            formula = "value ~ C(group)"
            model = smf.mixedlm(formula, model_df, groups=model_df["mouse_id"])
            result = model.fit()
            
            # Extract p-value for group effect
            group_param = [p for p in result.pvalues.index if 'group' in p.lower()]
            if group_param:
                p_val = result.pvalues[group_param[0]]
            else:
                raise KeyError(f"Group effect not found in model. Parameters: {list(result.params.index)}")
            
            print(f"  Mixed-effects model p-value for {col}: {p_val:.4f}")
            
            # Store p-value and determine significance
            p_values[col] = p_val
            significance[col] = sig_stars(p_val, alpha=alpha)
            
        except Exception as e:
            print(f"Error fitting mixed model for {col}: {e}")
            
            # Fallback to independent t-test with mouse-averaged data
            from scipy import stats
            
            # Group data by mouse ID and calculate means for each mouse
            mouse_means1 = {}
            mouse_means2 = {}
            
            for i, mouse_id in enumerate(mouse_ids1):
                if mouse_id in mouse_means1:
                    mouse_means1[mouse_id].append(df1.iloc[i][col])
                else:
                    mouse_means1[mouse_id] = [df1.iloc[i][col]]
                    
            for i, mouse_id in enumerate(mouse_ids2):
                if mouse_id in mouse_means2:
                    mouse_means2[mouse_id].append(df2.iloc[i][col])
                else:
                    mouse_means2[mouse_id] = [df2.iloc[i][col]]
            
            # Calculate mean value for each mouse
            mouse_values1 = [np.mean(vals) for vals in mouse_means1.values()]
            mouse_values2 = [np.mean(vals) for vals in mouse_means2.values()]
            
            # Run t-test on mouse-averaged data
            t_stat, p_val = stats.ttest_ind(mouse_values1, mouse_values2, equal_var=False)
            p_values[col] = p_val
            significance[col] = sig_stars(p_val, alpha=alpha)
            
            print(f"  Falling back to mouse-averaged t-test for {col}, p = {p_val:.4f}")
            print(f"  Group sizes after averaging: {label1}={len(mouse_values1)}, {label2}={len(mouse_values2)}")
    
    return p_values, significance

