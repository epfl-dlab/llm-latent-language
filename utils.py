import json
import gzip
import _pickle as pickle
import logging
import yaml
import json
from logging import Logger
import re
from datetime import datetime
import os
import pytz
from matplotlib import pyplot as plt 
import seaborn as sns
from scipy.stats import bootstrap
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.size': 16
})

plt_params = {'linewidth': 2.2}

def plot_ci_plus_heatmap(data, heat, labels, 
                         color='blue', 
                         linestyle='-',
                         tik_step=10, 
                         method='gaussian', 
                         do_lines=True, 
                         do_colorbar=False, 
                         shift=0.5, 
                         nums = [.99, 0.18, 0.025, 0.6],
                         labelpad=10,
                         plt_params=plt_params):
    
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 10]}, figsize=(5, 3))
    if do_colorbar:
        fig.subplots_adjust(right=0.8) 
    plot_ci(ax2, data, labels, color=color, linestyle=linestyle, tik_step=tik_step, method=method, do_lines=do_lines, plt_params=plt_params)
    
    y = heat.mean(dim=0)
    x = np.arange(y.shape[0])+1

    extent = [x[0]-(x[1]-x[0])/2. - shift, x[-1]+(x[1]-x[0])/2. + shift, 0, 1]
    img =ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent, vmin=0, vmax=14)
    ax.set_yticks([])
    #ax.set_xlim(extent[0], extent[1])
    if do_colorbar:
        cbar_ax = fig.add_axes(nums)  # Adjust these values as needed
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label('entropy', rotation=90, labelpad=labelpad)  # Adjust label and properties as needed
    plt.tight_layout()
    return fig, ax, ax2

def process_axis(ax, ylabel_font=13, xlabel_font=13):
    ax.spines[['right', 'top']].set_visible(False)
    #ax.set_ylabel(ylabel, fontsize=ylabel_font)
    #ax.set_xlabel(xlabel, fontsize=xlabel_font)

def plot_ci(ax, data, label, color='blue', linestyle='-', tik_step=10, method='gaussian', do_lines=True, plt_params=plt_params):
    if do_lines:
        upper = max(round(data.shape[1]/10)*10+1, data.shape[1]+1)
        ax.set_xticks(np.arange(0, upper, tik_step))
        for i in range(0, upper, tik_step):
            ax.axvline(i, color='black', linestyle='--', alpha=0.2, linewidth=1)
    if method == 'gaussian':
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        data_ci = {
            'x' : np.arange(data.shape[1])+1,
            'y' : mean,
            'y_upper' : mean + (1.96/(data.shape[0]**0.5)) * std,
            'y_lower' : mean - (1.96/(data.shape[0]**0.5)) * std,
        }
    elif method == 'np':
        data_ci = {
            'x' : np.arange(data.shape[1])+1,
            'y' : np.quantile(data, 0.5, axis=0),
            'y_upper' : np.quantile(data, 0.95, axis=0),
            'y_lower' : np.quantile(data, 0.05, axis=0),
        }
    elif method == 'bootstrap':
        bootstrap_ci = bootstrap((data,), np.mean, confidence_level=0.95, method='percentile')
        data_ci = {
            'x' : np.arange(data.shape[1])+1,
            'y' : data.mean(axis=0),
            'y_upper' : bootstrap_ci.confidence_interval.high,
            'y_lower' : bootstrap_ci.confidence_interval.low,
        }

    else:
        raise ValueError('method not implemented')

    df = pd.DataFrame(data_ci)
    # Create the line plot with confidence intervals
    ax.plot(df['x'], df['y'], label=label, color=color, linestyle=linestyle, **plt_params)
    ax.fill_between(df['x'], df['y_lower'], df['y_upper'], color=color, alpha=0.3)
    process_axis(ax)

def yaml_to_dict(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)



def save_pickle(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_pickle(path):
    if path.endswith('gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'rb') as f:
        return pickle.load(f)


def printr(text):
    print(f'[running]: {text}')
    

def save_json(data: object, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def prepare_output_dir(base_dir: str = "./runs/") -> str:
    # create output directory based on current time (using zurich time zone)
    experiment_dir = os.path.join(
        base_dir, datetime.now(tz=pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def get_logger(output_dir) -> Logger:
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Log to file
    file_path = os.path.join(LOG_DIR, f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log')
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def get_api_key(fname, provider='azure', key=None):
    print(fname)
    try:
        with open(fname) as f:
            keys = json.load(f)[provider]
            if key is not None:
                api_key = keys[key]
            else:
                api_key = list(keys.values())[0]
    except Exception as e:
        print(f'error: unable to load {provider} api key {key} from file {fname} - {e}')
        return None

    return api_key

def read_json(path_name: str):
    with open(path_name, "r") as f:
        json_file = json.load(f)
    return json_file

def printv(msg, v=0, v_min=0, c=None, debug=False):
    # convenience print function
    if debug:
        c = 'yellow' if c is None else c
        v, v_min = 1, 0
        printc('\n\n>>>>>>>>>>>>>>>>>>>>>>START DEBUG\n\n', c='yellow')
    if (v > v_min) or debug:
        if c is not None:
            printc(msg, c=c)
        else:
            print(msg)
    if debug:
        printc('\n\nEND DEBUG<<<<<<<<<<<<<<<<<<<<<<<<\n\n', c='yellow')


def printc(x, c='r'):
    m1 = {'r': 'red', 'g': 'green', 'y': 'yellow', 'w': 'white',
          'b': 'blue', 'p': 'pink', 't': 'teal', 'gr': 'gray'}
    m2 = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'pink': '\033[95m',
        'teal': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m'
    }
    reset_color = '\033[0m'
    print(f'{m2.get(m1.get(c, c), c)}{x}{reset_color}')

def extract_dictionary(x):
    if isinstance(x, str):
        regex = r"{.*?}"
        match = re.search(regex, x, re.MULTILINE | re.DOTALL)
        if match:
            try:
                json_str = match.group()
                json_str = json_str.replace("'", '"')
                dict_ = json.loads(json_str)
                return dict_
            except Exception as e:
                print(f"unable to extract dictionary - {e}")
                return None

        else:
            return None
    else:
        return None
