import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set(font_scale=1.5)
sns.set_theme(style='white')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
from IPython.core.display import HTML, display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tiktoken  
from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from transformers import GPT2Model, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel
import re

import sys
sys.path.append('../')
from models.configs import Configs, set_seed
set_seed(1987)
#=============================================================================================#
#=============================================================================================#
#=============================================================================================#
EXTREME_LOWER_THRESHOLD = 22.24
EXTREME_UPPER_THRESHOLD = 54.36
HECTARE_TO_ACRE_SCALE = 2.471  # 2.2417
MAXIMUM_AXIS_VALUE = 75
WEEK_FOR_VIS = 1

PLOT_CMAP = 'viridis'
PLOT_MINCNT = 100
PLOT_VIS_FACE_COLOR = 'white'
PLOT_BOX_FACE_COLOR = 'grey'
PLOT_TEXT_FONTSIZE = 13

#=========================================================================#
#================================ Helpers ================================#
#=========================================================================#
def regression_metrics(ytrue, ypred):
    """Calculating the evaluation metric based on regression results:
    input:
            ytrue: 
            ypred:
            
    output:
            root mean square error(rmse)
            root square(r^2)
            mean absolute error(mae)
            mean absolute percent error(mape)
            mean of true yield
            mean of prediction yield
    """

    r_square   = r2_score(ytrue, ypred)
    mae        = mean_absolute_error(ytrue, ypred)
    rmse       = np.sqrt(mean_squared_error(ytrue, ypred))
    mape       = mean_absolute_percentage_error(ytrue, ypred)*100
    mean_ytrue = np.mean(ytrue)
    mean_ypred = np.mean(ypred)
    
    return [r_square , mae, rmse, mape, mean_ytrue, mean_ypred] 

def apply_percentile(row, target_percentile):
    # Check if row is array-like, if not, make it an array
    if not isinstance(row, np.ndarray):
        row = np.array([row])
    return np.percentile(row, target_percentile)

def safe_mean(x):
    if isinstance(x, np.ndarray):
        return np.mean(x)
    else:
        return x
    
def select_closest_percentile(target_key, percentile_dict):

    closest_key = min(percentile_dict.keys(), key=lambda x: abs(x - target_key))
    return percentile_dict[closest_key]

def find_percentile_of_target(target, values_list):
    """
    Find the percentile of a target value within a list of values.

    Args:
    target (float or int): The target value.
    values_list (list): A list of numeric values.

    Returns:
    float: The percentile of the target within the list.
    """
    sorted_values = sorted(values_list)

    less_than_target = sum(value <= target for value in sorted_values)

    percentile = 100 * less_than_target / len(values_list)

    return percentile

def _find_closest_value(my_list, target):
    closest_value = my_list[0]
    minimum_diff = abs(my_list[0] - target)
    for value in my_list:
        diff = abs(value - target)
        if diff < minimum_diff:
            closest_value = value
            minimum_diff = diff
    return closest_value

def _return_shorter_df(df, category: str):

    blocks = df.groupby(by='block')
    names, ytrue, ypred = [], [], []
    cultivar, x, y = [], [], []
    closests, percentiles = [], []

    for b, bdf in blocks:
        same_coords = bdf.groupby(['x', 'y'])
        
        for g, c in same_coords: 
            if c.shape[0] > 1:
                mean_ytrue = c['ytrue'].mean().astype('float32')
                list_pred = c['ypred_w15'].values.astype('float32')
                closest    = _find_closest_value(c['ypred_w15'].values, mean_ytrue)
                percentile = find_percentile_of_target(closest, c['ypred_w15'].values)
            else: 
                mean_ytrue = c['ytrue'].values[0].astype('float32')
                list_pred = c['ypred_w15'].values[0].astype('float32')
                closest    = list_pred
                percentile = 50


            names.append(b)
            cultivar.append(c.iloc[0]['cultivar'])
            x.append(c.iloc[0]['x'])
            y.append(c.iloc[0]['y'])
            ytrue.append(mean_ytrue)
            ypred.append(list_pred)
            closests.append(closest)
            percentiles.append(percentile)

    # Create DataFrame
    m_df = pd.DataFrame({
        'block': names,
        'cultivar': cultivar,
        'x': x, 
        'y': y,
        'ytrue': ytrue,
        'ypred_w15': ypred,
        'closest': closests,
        'p': percentiles,
    })

    if category == 'train':
        m_df['ytrue_rounded'] = np.floor(m_df['ytrue'])
        mean_percentile_per_bin = m_df.groupby('ytrue_rounded')['p'].mean()
        # Convert the Series to a dictionary
        mean_percentile_dict = mean_percentile_per_bin.to_dict()

        return mean_percentile_dict
    else:
        return m_df
    
def return_iter_test_dfs(dir):
    list_df = []
    for i in range(10):
        test_df = pd.read_csv(os.path.join(dir + f'_test_{i}.csv'), index_col=0)
        # test_df = return_modified_df(test_df)
        list_df.append(test_df)

    return list_df

def aggregate(src, scale):
    
    w = int(src.shape[0]/scale)
    h = int(src.shape[1]/scale)
    mtx = np.full((w, h), -1, dtype=np.float32)
    for i in range(w):
        for j in range(h):   
            mtx[i,j]=np.mean(src[i*scale:(i+1)*scale, j*scale:(j+1)*scale])                    
    return mtx        

def return_modified_df(test_df, cat: str):

    # short_test_df = _return_shorter_df(test_df, category= 'test')
    # mean_percentile_dict = _return_shorter_df(train_df, category= 'train')
    
    results = {
        'block': [], 'x': [], 'y': [], 'cultivar': [],
        'ytrue': [],
        **{f'ypred_w{i}': [] for i in range(1, 16)}
    }

    
    def process_group(name, group):
        mean_ytrue = group['ytrue'].mean().astype('float32')

        if mean_ytrue < 9:
            if cat == 'extreme': 
                percentile_value = 100
            elif cat == 'mean': 
                percentile_value = 50
        elif mean_ytrue >= 20:
            if cat == 'extreme': 
                percentile_value = 100
            elif cat == 'mean': 
                percentile_value = 50
        else:
            percentile_value = None 

        for i in range(1, 16):
            key = f'ypred_w{i}'
            if percentile_value is not None:
                result = np.percentile(group[key], percentile_value).astype('float32')
            else:
                result = group[key].max().astype('float32')
            results[key].append(result)

        results['block'].append(name)
        results['x'].append(group.iloc[0]['x'])
        results['y'].append(group.iloc[0]['y'])
        results['cultivar'].append(group.iloc[0]['cultivar'])
        results['ytrue'].append(mean_ytrue)

    
    for name, block in test_df.groupby('block'):
        for _, group in block.groupby(['x', 'y']):
            process_group(name, group)

    #         # Apply the safe_mean function to each row and then calculate the overall mean
    # mean_value = short_test_df['ypred_w15'].apply(safe_mean).mean()
    # closest_percentile = select_closest_percentile(mean_value, mean_percentile_dict)

    # short_test_df['final'] = short_test_df['ypred_w15'].apply(lambda x: apply_percentile(x, closest_percentile))
    # short_test_df
    return pd.DataFrame(results) #short_test_df#,

def calc_test_blocks_range_values(range: str):

    df = pd.read_csv('/data2/hkaman/Imbalance/EXPs/CNNs/EXP_00_lr001_wd05_drop30_vanilla/00_lr001_wd05_drop30_vanilla_test.csv')
    
    blocks_df = df.groupby(by = 'block')

    blocks_name, blocks_cutilvar, blocks_mean_value = [], [], []
    for b, bdf in blocks_df:
        blocks_name.append(b)
        blocks_cutilvar.append(bdf.iloc[0]['cultivar'])
        blocks_mean_value.append(bdf['ytrue'].mean())

    newdf = pd.DataFrame()
    newdf['block'] = blocks_name
    newdf['cultivar'] = blocks_cutilvar
    newdf['mean'] = blocks_mean_value


    low_range_blocks = newdf[newdf['mean'] < 9]
    common_blocks = newdf[(newdf['mean'] >= 9) & (newdf['mean'] < 22)]
    high_range_blocks = newdf[newdf['mean'] >= 22]

    
    if range == 'low': 
        for idx, row in low_range_blocks.iterrows():
            print(f"[C1] LOW EXTEME RANGE: block name = {int(row['block'])} | cultivar type {row['cultivar']} | mean yield value = {row['mean']*HECTARE_TO_ACRE_SCALE:.2f}")
    elif range == 'common':
        for idx, row in common_blocks.iterrows():
            print(f"[C2] COMMON: block name = {int(row['block'])} | cultivar type {row['cultivar']} | mean yield value = {row['mean']*HECTARE_TO_ACRE_SCALE:.2f}")
    elif range == 'high':
        for idx, row in high_range_blocks.iterrows():
            print(f"[C3] High EXTEME RANGE: block name = {int(row['block'])} | cultivar type {row['cultivar']} | mean yield value = {row['mean']*HECTARE_TO_ACRE_SCALE:.2f}")

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

class weight_vis():
    def __init__(self, 
                 method: str, 
                 lds_ks: int = 10, 
                 lds_sigma: int = 8,
                 dw_alpha: float = 3.9,
                 betha: float = 4):

        self.method = method
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        self.dw_alpha = dw_alpha
        self.betha = betha

        self.df = pd.read_csv('/data2/hkaman/Imbalance/EXPs/CNNs/EXP_02_lr001_wd05_LDSinv_10_8/coords/train.csv')

    def plot(self):
        ytrue = self._return_ytrue()
        if self.method == 'lds':
            weights = self._return_lds_weights(ytrue)
        elif self.method == 'cb':
            weights = self._return_cb_weights(ytrue)
        elif self.method == 'dw':
            weights = self._return_dw_weights(ytrue)
        elif self.method == 'ours':
            weights = self._return_extreme_weights(ytrue)

        weight_means, ytrue_counts = self._return_samples_weight_per_bins(ytrue, weights)    
        
        self._plot(ytrue_counts, weight_means)

        return weight_means

    def crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        crop_src = src[:, xcoord:xcoord + 16, ycoord:ycoord + 16, :]
        return crop_src 
    
    def _return_extreme_weights(self, ytrue):
        all_mean_value = np.mean(ytrue)
        # Perform Kernel Density Estimation
        density_lds = self._return_lds_weights(ytrue)

        # density_lds =  RWSampler.cb_prepare_weights(ytrue, 
        #                                 lds_kernel = 'gaussian', 
        #                                 lds_ks = self.lds_ks, 
        #                                 lds_sigma = self.lds_sigma, 
        #                                 betha = 3.9)
        
        density_lds = density_lds/np.max(density_lds)
        weights = ((density_lds) * ((ytrue - all_mean_value)**2)) + 1
        weights = np.array(weights, dtype = np.float32)

        return weights

    def _return_lds_weights(self, ytrue):
        weights = RWSampler.lds_prepare_weights(ytrue, 'inverse', 
                            max_target = 30, 
                            lds = True, 
                            lds_kernel = 'gaussian', 
                            lds_ks = self.lds_ks, 
                            lds_sigma = self.lds_sigma)
        
        weights = np.array(weights, dtype = np.float32)
        return weights
    
    def _return_cb_weights(self, ytrue):

        weights =  RWSampler.cb_prepare_weights(ytrue, 
                                                lds_kernel = 'gaussian', 
                                                lds_ks = self.lds_ks, 
                                                lds_sigma = self.lds_sigma, 
                                                betha = self.betha)
        weights = np.array(weights, dtype = np.float32)
        return weights  
    
    def _return_dw_weights(self, ytrue):

        weights = RWSampler.TargetRelevance(ytrue, alpha = self.dw_alpha).__call__(ytrue)
        assert not np.isnan(weights).any()

        # weights = np.where(weights >= 1, weights, 1)

        return weights

    def _return_ytrue(self):
        masks = None
        for idx, row in self.df.iterrows():
            xcoord     = row['X'] 
            ycoord     = row['Y'] 
            label_path = row['LABEL_PATH'] 
            mask  = self.crop_gen(label_path, xcoord, ycoord) 
            mask  = np.swapaxes(mask, -1, 0)

            if masks is None: 
                masks = mask
            else: 
                masks = np.concatenate([masks, mask], axis = 0)

        reshaped_masks = np.reshape(masks, (masks.shape[0]*masks.shape[1]*masks.shape[2]))

        return reshaped_masks
    
    def _plot(self, ytrue_counts, weights):

        fig, axs = plt.subplots(1, 1, figsize=(12, 5))

        sns.set_style("whitegrid", {'axes.grid': False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace=0.01)

        bins_value = np.floor(np.arange(1, 75, 2.5)).astype(int)

        ax1 = sns.barplot(x=bins_value, y=ytrue_counts, color= sns.color_palette()[0], ax=axs) #sns.color_palette()[0]
        axs01 = axs.twinx()
        ax01 = sns.barplot(x=bins_value, y=weights, color='red', alpha=0.6, ax=axs01)

        handles = [mpatches.Patch(facecolor=sns.color_palette()[0], label='Number of Samples'),
                mpatches.Patch(facecolor='red', label='Weight')]
        axs.legend(handles=handles, loc='upper right')

        ax1.set(ylabel='Number of Samples')
        ax01.set(ylabel='Weight')
        ax1.set(xlabel='bin')

        plt.show()
    
    def _return_samples_weight_per_bins(self, masks, weights):
        ytrue_counts, weight_means = [], []
        for i in range(30):
            # Create a mask for the bin
            bin_mask = (masks > i) & (masks <= (i + 1)) if i > 0 else (masks >= i) & (masks <= (i + 1))

            # Apply the mask to count and calculate mean
            ytrue_count = np.count_nonzero(bin_mask)
            weight_mean = np.mean(weights[bin_mask])

            ytrue_counts.append(ytrue_count)
            weight_means.append(weight_mean)

        return weight_means, ytrue_counts
#=========================================================================#
#================================ Classes ================================#
#=========================================================================#

class performance():
    def __init__(self, 
                 exp_name: str):
        self.exp_name = exp_name

        self.exp_output_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep/' + 'EXP_' + exp_name 

        self.train_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_train.csv'), index_col=0)
        self.train_df = self.train_df[self.train_df['ypred_w1'] > 0]

        self.valid_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_valid.csv'), index_col=0)
        self.valid_df = self.valid_df[self.valid_df['ypred_w1'] > 0]

        self.test_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_test.csv'), index_col=0)
        self.test_df = self.test_df[self.test_df['ypred_w1'] > 0]

        # self.iter_test_list_df = return_iter_test_dfs(os.path.join(self.exp_output_dir, exp_name))
        # self.modified_quantile_test_df = return_modified_df(self.test_df)
    def scatter_plot(self):

        fig = plt.figure(figsize=(21, 7))
        gs  = gridspec.GridSpec(1, 3)

        train_plot = self._plot_scatter(self.train_df)
        valid_plot = self._plot_scatter(self.valid_df)
        test_plot = self._plot_scatter(self.test_df)

        mg0 = SeabornFig2Grid(train_plot, fig, gs[0])
        mg1 = SeabornFig2Grid(valid_plot, fig, gs[1])
        mg2 = SeabornFig2Grid(test_plot, fig, gs[2])

        gs.tight_layout(fig)
        plt.show()
    
    def time_series_plot(self):
        
        Weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
        
        results = pd.DataFrame()
        
        # test_r2_mean, test_mae_mean, test_rmse_mean, test_mape_mean = [], [], [], []
        # test_r2_std, test_mae_std, test_rmse_std, test_mape_std = [], [], [], []
        test_r2_temp, test_mae_temp, test_rmse_temp, test_mape_temp = [], [], [], []
        for w in range(1, 16, 1): 
        
            # for i in range(10):
            test_r2, test_mae, test_rmse, test_mape, _, _ = regression_metrics(self.test_df['ytrue']* HECTARE_TO_ACRE_SCALE, 
                                                                                self.test_df[f'ypred_w{w}']* HECTARE_TO_ACRE_SCALE)
            test_r2_temp.append(test_r2)
            test_mae_temp.append(test_mae)
            test_rmse_temp.append(test_rmse)
            test_mape_temp.append(test_mape)
                
            
            # test_r2_mean.append(np.mean(test_r2_temp))
            # test_mae_mean.append(np.mean(test_mae_temp))
            # test_rmse_mean.append(np.mean(test_rmse_temp))
            # test_mape_mean.append(np.mean(test_mape_temp))
            
            # test_r2_std.append(np.std(test_r2_temp))
            # test_mae_std.append(np.std(test_mae_temp))
            # test_rmse_std.append(np.std(test_rmse_temp))
            # test_mape_std.append(np.std(test_mape_temp))
        
        results['weeks'] = Weeks
        results['R2_mean'] = test_r2_temp
        results['MAE_mean'] = test_mae_temp
        results['RMSE_mean'] = test_rmse_temp
        results['MAPE_mean'] = test_mape_temp
        # results['R2_std'] = test_r2_std
        # results['MAE_std'] = test_mae_std
        # results['RMSE_std'] = test_rmse_std
        # results['MAPE_std'] = test_mape_std

        fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
        metrics = [('R2', 'R2'), ('RMSE', 'RMSE'), ('MAE', 'MAE (t/ha)'), ('MAPE', 'MAPE (%)')]
        axs = axs.flatten() 

        for ax, (metric, label) in zip(axs, metrics):
            ax.plot(results["weeks"], results[f'{metric}_mean'], "-d")
            # ax.fill_between(results["weeks"], results[f'{metric}_mean'] - results[f'{metric}_std'], 
            #                 results[f'{metric}_mean'] + results[f'{metric}_std'], alpha=.2)
            ax.set_ylabel(label)
            ax.set_facecolor('white')
            plt.setp(ax.spines.values(), color='k')

        axs[2].tick_params(axis='x', rotation=45) 
        axs[-1].tick_params(axis='x', rotation=45) 
        axs[-1].legend(loc="upper right")
        plt.show()
        None

        # return results
    
    def mape_per_yield_range(self, th1: int, th2: int):

        test_df_ytrue = self.test_df['ytrue'].values * HECTARE_TO_ACRE_SCALE

        len_C1 = len(test_df_ytrue[np.where((test_df_ytrue >= 0) & (test_df_ytrue < th1))])
        len_C2 = len(test_df_ytrue[np.where((test_df_ytrue >= th1) & (test_df_ytrue < th2))])
        len_C3 = len(test_df_ytrue[np.where(test_df_ytrue >= th2)])
        #
        true_labels = self.test_df['ytrue'].values * HECTARE_TO_ACRE_SCALE
        pred_labels = self.test_df['ypred_w1'].values * HECTARE_TO_ACRE_SCALE

        #if i < th1: 
        true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels < th1))]
        pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels < th1))]

        #elif (i >= th1) & (i < th2):
        true_label_C2 = true_labels[np.where((true_labels >= th1) & (true_labels < th2))]
        pred_label_C2 = pred_labels[np.where((true_labels >= th1) & (true_labels < th2))]

        #elif i >= th2: 
        true_label_C3 = true_labels[np.where(true_labels >= th2)]
        pred_label_C3 = pred_labels[np.where(true_labels >= th2)]

        All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
        # print(f"C1 num samples: {len_C1} | C2 num samples: {len_C2} | C3 num samples: {len_C3}")
        C1_R2, C1_MAE, C1_RMSE, C1_MAPE, _, _ = regression_metrics(true_label_C1, pred_label_C1)
        C2_R2, C2_MAE, C2_RMSE, C2_MAPE, _, _ = regression_metrics(true_label_C2, pred_label_C2)
        C3_R2, C3_MAE, C3_RMSE, C3_MAPE, _, _ = regression_metrics(true_label_C3, pred_label_C3)

        # print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
        print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | C1: MAE = {C1_MAE:.2f}, MAPE = {C1_MAPE:.2f} | C2: MAE = {C2_MAE:.2f}, MAPE = {C2_MAPE:.2f} | C3: MAE = {C3_MAE:.2f}, MAPE = {C3_MAPE:.2f}")
        print(f"=========================================================================================================================")
        # return [C1_MAE, C1_MAPE, C2_MAE, C2_MAPE, C3_MAE, C3_MAPE]

    def sensivity_mape_per_yield_range(self, keyword: str, th1: int, th2: int):
        cleaned_keyword = re.sub(r'[^a-zA-Z0-9_]', '', keyword)
        test_df_ytrue = pd.read_csv(os.path.join(self.exp_output_dir, self.exp_name + '_test_'+ cleaned_keyword +'.csv'))
        test_df_ytrue = test_df_ytrue[test_df_ytrue['ypred_w1'] > 0]
        # test_df_ytrue = self.test_df['ytrue'].values * HECTARE_TO_ACRE_SCALE

        # len_C1 = len(test_df_ytrue[np.where((test_df_ytrue >= 0) & (test_df_ytrue < th1))])
        # len_C2 = len(test_df_ytrue[np.where((test_df_ytrue >= th1) & (test_df_ytrue < th2))])
        # len_C3 = len(test_df_ytrue[np.where(test_df_ytrue >= th2)])
        #
        true_labels = test_df_ytrue['ytrue'].values * HECTARE_TO_ACRE_SCALE
        pred_labels = test_df_ytrue['ypred_w1'].values * HECTARE_TO_ACRE_SCALE

        #if i < th1: 
        true_label_C1 = true_labels[np.where((true_labels >= 0) & (true_labels < th1))]
        pred_label_C1 = pred_labels[np.where((true_labels >= 0) & (true_labels < th1))]

        #elif (i >= th1) & (i < th2):
        true_label_C2 = true_labels[np.where((true_labels >= th1) & (true_labels < th2))]
        pred_label_C2 = pred_labels[np.where((true_labels >= th1) & (true_labels < th2))]

        #elif i >= th2: 
        true_label_C3 = true_labels[np.where(true_labels >= th2)]
        pred_label_C3 = pred_labels[np.where(true_labels >= th2)]

        All_R2, All_MAE, All_RMSE, All_MAPE, _, _ = regression_metrics(true_labels, pred_labels)
        # print(f"C1 num samples: {len_C1} | C2 num samples: {len_C2} | C3 num samples: {len_C3}")
        C1_R2, C1_MAE, C1_RMSE, C1_MAPE, _, _ = regression_metrics(true_label_C1, pred_label_C1)
        C2_R2, C2_MAE, C2_RMSE, C2_MAPE, _, _ = regression_metrics(true_label_C2, pred_label_C2)
        C3_R2, C3_MAE, C3_RMSE, C3_MAPE, _, _ = regression_metrics(true_label_C3, pred_label_C3)

        # print(f"C1 is yield value between 0 and {th1}, C2 is yield value between {th1} and {th2}, and C3 is yield value bigger than {th2}")
        print(f"All: MAE = {All_MAE:.2f}, MAPE = {All_MAPE:.2f} | C1: MAE = {C1_MAE:.2f}, MAPE = {C1_MAPE:.2f} | C2: MAE = {C2_MAE:.2f}, MAPE = {C2_MAPE:.2f} | C3: MAE = {C3_MAE:.2f}, MAPE = {C3_MAPE:.2f}")
        print(f"=========================================================================================================================")
        # return [C1_MAE, C1_MAPE, C2_MAE, C2_MAPE, C3_MAE, C3_MAPE]

    def plot_sensivity(self, keyword: str, th1: int, th2: int):
        cleaned_keyword = re.sub(r'[^a-zA-Z0-9_]', '', keyword)
        test_df_ytrue_modified = pd.read_csv(os.path.join(self.exp_output_dir, self.exp_name + '_test_'+ cleaned_keyword +'.csv'))
        test_df_ytrue_modified = test_df_ytrue_modified[test_df_ytrue_modified['ypred_w1'] > 0]
        true_labels_modified = test_df_ytrue_modified['ytrue'].values * HECTARE_TO_ACRE_SCALE
        pred_labels_modified = test_df_ytrue_modified['ypred_w1'].values * HECTARE_TO_ACRE_SCALE


        true_labels_org = self.test_df['ytrue'].values * HECTARE_TO_ACRE_SCALE
        pred_labels_org = self.test_df['ypred_w1'].values * HECTARE_TO_ACRE_SCALE



        #if i < th1: 
        true_label_C1_org= true_labels_org[np.where((true_labels_org >= 0) & (true_labels_org < th1))]
        pred_label_C1_org = pred_labels_org[np.where((true_labels_org >= 0) & (true_labels_org < th1))]
        true_label_C1_modified = true_labels_modified[np.where((true_labels_modified >= 0) & (true_labels_modified < th1))]
        pred_label_C1_modified = pred_labels_modified[np.where((true_labels_modified >= 0) & (true_labels_modified < th1))]

        #elif (i >= th1) & (i < th2):
        true_label_C2_org = true_labels_org[np.where((true_labels_org >= th1) & (true_labels_org < th2))]
        pred_label_C2_org = pred_labels_org[np.where((true_labels_org >= th1) & (true_labels_org < th2))]
        true_label_C2_modified = true_labels_modified[np.where((true_labels_modified >= th1) & (true_labels_modified < th2))]
        pred_label_C2_modified = pred_labels_modified[np.where((true_labels_modified >= th1) & (true_labels_modified < th2))]

        #elif i >= th2: 
        true_label_C3_org = true_labels_org[np.where(true_labels_org >= th2)]
        pred_label_C3_org = pred_labels_org[np.where(true_labels_org >= th2)]
        true_label_C3_modified = true_labels_modified[np.where(true_labels_modified >= th2)]
        pred_label_C3_modified = pred_labels_modified[np.where(true_labels_modified >= th2)]

            # Plotting with seaborn
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))  # 1 row, 3 columns

        # C1 plot
        sns.kdeplot(pred_label_C1_org, fill=True, label='Original', ax=axes[0])
        sns.kdeplot(pred_label_C1_modified, fill=True, label='Modified', ax=axes[0])
        mean_shift_C1 = np.mean(pred_label_C1_modified) - np.mean(pred_label_C1_org)
        axes[0].legend(title=f"Mean shift: {mean_shift_C1:.2f}")
        axes[0].set_title(f'C1: true labels < {th1}')

        # C2 plot
        sns.kdeplot(pred_label_C2_org, fill=True, label='Original', ax=axes[1])
        sns.kdeplot(pred_label_C2_modified, fill=True, label='Modified', ax=axes[1])
        mean_shift_C2 = np.mean(pred_label_C2_modified) - np.mean(pred_label_C2_org)
        axes[1].legend(title=f"Mean shift: {mean_shift_C2:.2f}")
        axes[1].set_title(f'C2: {th1} <= true labels < {th2}')

        # C3 plot
        sns.kdeplot(pred_label_C3_org, fill=True, label='Original', ax=axes[2])
        sns.kdeplot(pred_label_C3_modified, fill=True, label='Modified', ax=axes[2])
        mean_shift_C3 = np.mean(pred_label_C3_modified) - np.mean(pred_label_C3_org)
        axes[2].legend(title=f"Mean shift: {mean_shift_C3:.2f}")
        axes[2].set_title(f'C3: true labels >= {th2}')

        # Display the plots
        plt.tight_layout()
        plt.show()




    def mape_per_bin_plot(self):
        fig, axs = plt.subplots(1, 1, figsize=(16, 4))

        sns.set_style("whitegrid", {'axes.grid': False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace=0.01)
        self.modified_quantile_test_df = self.test_df
        MAPE_Errors, counts = [], []
        for i in range(1, 71, 2):
            if i == 1:
                count = self.modified_quantile_test_df.loc[(self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
                Data = self.test_df.loc[(self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
            elif i >= 68:  
                count = self.modified_quantile_test_df.loc[(self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i)]
                Data = self.test_df.loc[(self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i)]
            else:

                count = self.modified_quantile_test_df.loc[(self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i) & (self.modified_quantile_test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
                Data = self.test_df.loc[(self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE >= i) & (self.test_df['ytrue'] * HECTARE_TO_ACRE_SCALE < (i + 2))]
            
            counts.append(len(count))
            if len(Data) == 0 or (i<=10):
                MAPE = 0    
            else:
                MAPE = mean_absolute_percentage_error(Data['ytrue'], Data['ypred_w15'])
            if MAPE > 1:
                MAPE = 0
            MAPE_Errors.append(MAPE * 100)

        # pearson_value = pearsonr(MAPE_Errors, counts)[0]
        bins_value = np.arange(1, 71, 2)  # Adjusted bins value

        ax1 = sns.barplot(x=bins_value, y=counts, color=sns.color_palette()[0], ax=axs)
        axs01 = axs.twinx()
        ax01 = sns.barplot(x=bins_value, y=MAPE_Errors, color=sns.color_palette()[3], alpha=0.5, ax=axs01)

        handles = [mpatches.Patch(facecolor=sns.color_palette()[0], label='Number of Samples'),
                mpatches.Patch(facecolor=sns.color_palette()[3], label='MAPE Error (%)')]
        axs.legend(handles=handles, loc='upper right')

        ax1.set(ylabel='Number of Samples')
        ax01.set(ylabel='MAPE Error (%)')
        ax01.set_ylim(0, 100)
        ax1.set(xlabel='Yield Value (t/ha)')
        # plt.title(r"Pearson Correlation: {:.2f}".format(pearson_value))

        for j, m in enumerate(MAPE_Errors):        
            ax01.text(j ,m, "{:.2f}".format(m), color='k', position = (j - 0.15, m + 1.5), fontsize = 'small', rotation=90)

        plt.show()

    def _plot_scatter(self, df):

        week_pred = 'ypred_w' + str(WEEK_FOR_VIS)

        data = df[['ytrue', week_pred]].rename(columns={week_pred: "ypred"})

        true_values = data['ytrue'] * HECTARE_TO_ACRE_SCALE
        pred_values = data['ypred'] * HECTARE_TO_ACRE_SCALE

        r2, mae, rmse, mape, _, _ = regression_metrics(true_values, pred_values)

        g = sns.jointplot(x = true_values, 
                          y = pred_values, 
                          kind = "hex", 
                          height = 8, 
                          ratio = 4,
                          xlim = [0, MAXIMUM_AXIS_VALUE], ylim=[0, MAXIMUM_AXIS_VALUE], 
                          extent = [0, MAXIMUM_AXIS_VALUE, 0, MAXIMUM_AXIS_VALUE], 
                          gridsize = 100,
                          cmap = PLOT_CMAP, 
                          mincnt = PLOT_MINCNT, 
                          joint_kws = {"facecolor": PLOT_VIS_FACE_COLOR})

        for patch in g.ax_marg_x.patches:
            patch.set_facecolor(PLOT_BOX_FACE_COLOR)

        for patch in g.ax_marg_y.patches:
            patch.set_facecolor(PLOT_BOX_FACE_COLOR)

        g.ax_joint.plot([0, MAXIMUM_AXIS_VALUE], [0, MAXIMUM_AXIS_VALUE], '--r', linewidth=2)

        plt.xlabel('Measured (t/ha)', fontsize = 16)
        plt.ylabel('Predicted (t/ha)', fontsize = 16)
        plt.grid(False)

        scores = (r'R^2={:.2f}' + '\n' + r'MAE={:.2f} (t/ha)' + '\n' + r'RMSE={:.2f} (t/ha)' + '\n' + r'MAPE={:.2f} %').format(
            r2, mae, rmse, mape)

        plt.text(1, 74.7, scores, bbox=dict(facecolor = PLOT_VIS_FACE_COLOR, edgecolor = PLOT_BOX_FACE_COLOR, boxstyle = 'round, pad=0.2'),
                fontsize = 16, ha='left', va = 'top')

        return g

class multi_model_timeseries_plot():
    def __init__(self):
        self.weeks = ['Apr 01', 'Apr 08', 'Apr 17', 'Apr 26', 'May 05', 'May 15', 'May 21', 'May 30', 'Jun 10', 'Jun 16', 'Jun 21', 'Jun 27', 'Jul 02', 'Jul 09', 'Jul 15']
    
        model_files_dict = self._return_full_df_names()
        self.full_df = self._return_full_df(model_files_dict)
        

    def plot(self):
        custom_labels_dict = {
                'EXP_00_lr001_wd05_drop30_vanilla': 'Vanilla',
                'EXP_01_lr001_wd05_resampling': 'CSR',
                'EXP_02_lr001_wd05_drop30_focalr': 'Focal-R',
                'EXP_03_lr001_wd05_LDSinv_10_8': 'LDS',
                'EXP_04_lr001_wd05_DW_3.9': 'Dense Weight',
                'EXP_05_lr001_wd05_CB_3': 'Class Balanced',
                'EXP_06_lr001_wd05_ExW': 'Extreme Weight',
                'EXP_07_lr001_wd05_drop30_yz': 'Yield Zone',
                'EXP_08_lr001_wd05_drop30_resampling_yz': 'Extreme Weight + Yield Zone',
            }
        
        fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
        metrics = [('R2', 'R2'), ('RMSE', 'RMSE'), ('MAE', 'MAE (t/ha)'), ('MAPE', 'MAPE (%)')]
        axs = axs.flatten() 
        # custom_labels = ['Vanilla', 'CSR', 'LDS', 'Class Balanced', 'Dense Weight', 'CSR_Yield Zone']  
        markers = ['o', 'v', '^', '<', '>', 's', '^', 'o', 'v']  
        models = self.full_df['model'].unique()


        for ax, (metric, label) in zip(axs, metrics):
            for model, marker in zip(models, markers):
                custom_label = custom_labels_dict.get(model)
                sns.lineplot(x="weeks", y=f'{metric}_mean', data=self.full_df[self.full_df['model'] == model], 
                            ax=ax, marker=marker, linewidth = 2.5, label=custom_label)
            ax.set_ylabel(label, fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.set_facecolor('white')
            plt.setp(ax.spines.values(), color='k')
            if ax == axs[-1]:  # Show legend only for the last plot
                ax.legend(title='Model', loc="upper right", fontsize=10)
            else:
                ax.legend().remove()

        axs[2].tick_params(axis='x', labelsize=14, rotation=45) 
        axs[3].tick_params(axis='x', labelsize=14, rotation=45)
        plt.show()

    def _return_full_df(self, dict):

        full_df = pd.DataFrame()
        
        list_df = []
        test_r2, test_mae, test_rmse, test_mape = [], [], [], []
        test_r2, test_mae, test_rmse, test_mape = [], [], [], []

        for key, values in dict.items():
            model_df = pd.DataFrame()
            test_r2_mean, test_mae_mean, test_rmse_mean, test_mape_mean = [], [], [], []
            test_r2_std, test_mae_std, test_rmse_std, test_mape_std = [], [], [], []
            test_r2_temp, test_mae_temp, test_rmse_temp, test_mape_temp = [], [], [], []
            df = pd.read_csv(values[0], index_col=0)
            for w in range(1, 16, 1): 
                # for file in values:
                test_r2, test_mae, test_rmse, test_mape, _, _ = regression_metrics(df['ytrue']* HECTARE_TO_ACRE_SCALE, 
                                                                                    df[f'ypred_w{w}']* HECTARE_TO_ACRE_SCALE)
                
                test_r2_temp.append(test_r2)
                test_mae_temp.append(test_mae)
                test_rmse_temp.append(test_rmse)
                test_mape_temp.append(test_mape)
                    
                # test_r2_mean.append(np.mean(test_r2_temp))
                # test_mae_mean.append(np.mean(test_mae_temp))
                # test_rmse_mean.append(np.mean(test_rmse_temp))
                # test_mape_mean.append(np.mean(test_mape_temp))
                
                # test_r2_std.append(np.std(test_r2_temp))
                # test_mae_std.append(np.std(test_mae_temp))
                # test_rmse_std.append(np.std(test_rmse_temp))
                # test_mape_std.append(np.std(test_mape_temp))

            model_df['model'] = 15*[key]
            model_df['weeks'] = self.weeks
            model_df['R2_mean'] = test_r2_temp
            model_df['MAE_mean'] = test_mae_temp
            model_df['RMSE_mean'] = test_rmse_temp
            model_df['MAPE_mean'] = test_mape_temp
            # model_df['R2_std'] = test_r2_std
            # model_df['MAE_std'] = test_mae_std
            # model_df['RMSE_std'] = test_rmse_std
            # model_df['MAPE_std'] = test_mape_std

            list_df.append(model_df)

        full_df = pd.concat(list_df)

        return full_df

    def _return_full_df_names(self):

        base_dir = '/data2/hkaman/Imbalance/EXPs/CNNs'  # Replace with the actual path to your 'CNN' folder
        model_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

        model_files_dict = {}

        for model in model_dirs:
            model_path = os.path.join(base_dir, model)
            csv_files = sorted(glob.glob(os.path.join(model_path, '*test.csv')))
            model_files_dict[model] = csv_files
        
        return model_files_dict

class timeseries_spatial_variability():
    def __init__(
            self, 
            exp_name: str,
    ):

        self.exp_output_dir = '/data2/hkaman/Imbalance/EXPs/CNNs/' + 'EXP_' + exp_name 
        # train_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_train.csv'), index_col=0)
        test_df = pd.read_csv(os.path.join(self.exp_output_dir, exp_name + '_test_sp.csv'), index_col=0)
        self.test_df = return_modified_df(test_df, cat = 'extreme')
        # print(self.test_df.shape)
    def multiscale_plot(self, block_name: str, year:int,  min_v: None, max_v: None):

        ytrue10, list_ypred = self.return_rebuild_block_matrix(block_name, year = year)
        
        # 20x20
        ytrue20 = aggregate(ytrue10, 2)
        ypred_20_w15  = aggregate(list_ypred[-1], 2)

        # 40x40
        ytrue40 = aggregate(ytrue10, 4)
        ypred_40_w15  = aggregate(list_ypred[-1], 4)

        # 60x60
        ytrue60 = aggregate(ytrue10, 6)
        ypred_60_w15  = aggregate(list_ypred[-1], 6)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        im_true = axs[0, 0].imshow(ytrue10, vmin=min_v, vmax=max_v)
        axs[0, 0].set_title('Yield Observation (10m)', fontsize=16)
        # axs[0, 0].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_true,  cax=cax)
        im_true.set_clim(min_v, max_v)

        img = axs[0, 1].imshow(list_ypred[-1], vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue10, list_ypred[-1])
        axs[0, 1].set_title(f'Yield Prediction Week 15: \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[0, 1].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)
        
        # distribution plot: 
        ytrue10_vector = ytrue10.flatten()
        ytrue10_vector = ytrue10_vector[ytrue10_vector != -1]

        ypred10_vector = list_ypred[-1].flatten()
        ypred10_vector = ypred10_vector[ypred10_vector != -1]

        ypred20_vector = ypred_20_w15.flatten()
        ypred20_vector = ypred20_vector[ypred20_vector != -1]

        ypred40_vector = ypred_40_w15.flatten()
        ypred40_vector = ypred40_vector[ypred40_vector != -1]

        ypred60_vector = ypred_60_w15.flatten()
        ypred60_vector = ypred60_vector[ypred60_vector != -1]


        data = {
        'Yield (t/ha)': np.concatenate([ytrue10_vector, ypred10_vector, ypred20_vector, ypred40_vector, ypred60_vector]),
        'Matrix': ['Yield Observation']*len(ytrue10_vector) + ['Yield Prediction (10m)']*len(ypred10_vector) + ['Yield Prediction (20m)']*len(ypred20_vector) + ['Yield Prediction (40m)']*len(ypred40_vector) + ['Yield Prediction (60m)']*len(ypred60_vector) 
            }
        df = pd.DataFrame(data)
        
        categories = df['Matrix'].unique()
        for category in categories:
            sns.kdeplot(df[df['Matrix'] == category], x='Yield (t/ha)', fill=True, ax=axs[0, 2], label=category)
        axs[0, 2].legend(loc='upper left')


        img = axs[1, 0].imshow(ypred_20_w15, vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue20, ypred_20_w15)
        axs[1, 0].set_title(f'Yield Prediction Week 15 (20m): \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[1, 0].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)

        img = axs[1, 1].imshow(ypred_40_w15, vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue40, ypred_40_w15)
        axs[1, 1].set_title(f'Yield Prediction Week 15 (40m): \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[1, 1].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[1, 1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)
    
        img = axs[1, 2].imshow(ypred_60_w15, vmin=min_v, vmax=max_v)
        _, test_mae, _, test_mape, _, _ = regression_metrics(ytrue60, ypred_60_w15)
        axs[1, 2].set_title(f'Yield Prediction Week 15 (60m): \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=16)
        # axs[1, 2].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[1, 2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img,  cax=cax)
        img.set_clim(min_v, max_v)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.25, wspace=0.25)

    def timeseries_plot(self, block_name:str, year: int,  min_v: None, max_v: None):

        
        list_ytrue, list_ypred = self.return_rebuild_block_matrix(block_name, year = year)


        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(4, 4, figsize=(24, 24))

   
        im_true = axs[0, 0].imshow(list_ytrue, vmin=min_v, vmax=max_v)
        axs[0, 0].set_title('Yield Observation', fontsize=18)
        axs[0, 0].axis('off')  # Turn off axis
        divider = make_axes_locatable(axs[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_true,  cax=cax)
        im_true.set_clim(min_v, max_v)


        # Plotting predicted images
        for i in range(15):
            row, col = divmod(i + 1, 4)
            img = axs[row, col].imshow(list_ypred[i], vmin=min_v, vmax=max_v)
            _, test_mae, _, test_mape, _, _ = regression_metrics(list_ytrue, list_ypred[i])
            axs[row, col].set_title(f'Yield Prediction Week {i+1}: \nMAE (t/ha) = {test_mae:.2f}, MAPE = {test_mape:.2f}', fontsize=18)
            axs[row, col].axis('off')  # Turn off axis
            divider = make_axes_locatable(axs[row, col])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(img,  cax=cax)
            img.set_clim(min_v, max_v)

        # Adjust axes visibility for the last row and first column
        rows = [0, 1, 2]
        for i in rows:
            for ax in axs[i, :]:
                ax.axis('on')
                ax.set_xticks([])
                #ax.set_xlabel('X-axis Label')  # Set your x-axis label here
        columns = [1, 2, 3]
        for c in columns:
            for ax in axs[:, c]:
                ax.axis('on')
                ax.set_yticks([])
                #ax.set_ylabel('Y-axis Label')  # Set your y-axis label here
        axs[3, 0].axis('on')
        plt.tight_layout()
        plt.show()

    def plot(self, block_name:str, year: int, min_v: None, max_v: None):

        list_ytrue, list_ypred = self.return_rebuild_block_matrix(block_name, year = year)

        num_years = len(list_ytrue)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(1, 4, figsize = (24, 6))

        # for i in range(num_years):
        img1 = axs[0].imshow(list_ytrue)
        axs[0].set_title('Yield Observation', fontsize = 18)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(img1,  cax=cax)
        img1.set_clim(min_v, max_v)

        img2 = axs[1].imshow(list_ypred[-1])
        axs[1].set_title('Yield Prediction (Week 15)', fontsize = 18)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img2, cax=cax)
        img2.set_clim(min_v, max_v)
        axs[1].get_yaxis().set_visible(False)

        _, test_mae, _, test_mape, _, _ = regression_metrics(list_ytrue[list_ytrue !=- 1], list_ypred[-1][list_ypred[-1] != -1])


        mae_map, mape_map = self.image_mae_mape_map(list_ytrue, list_ypred[-1])

        img3 = axs[2].imshow(mae_map, cmap = 'viridis') #, cmap = 'magma'
        axs[2].set_title(f'MAE Map (t/ha) = {test_mae:.2f}', fontsize = 18)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img3, cax=cax)
        img3.set_clim(-1, 10)
        axs[2].get_yaxis().set_visible(False)

        img4 = axs[3].imshow(mape_map, cmap = 'viridis') #
        axs[3].set_title(f'MAPE Map (%) = {test_mape:.2f}', fontsize = 18)
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar3 =fig.colorbar(img4, cax=cax)
        img4.set_clim(-5, 20)
        axs[3].get_yaxis().set_visible(False)

        fig.subplots_adjust(hspace=0.01, wspace=0.01) 
        fig.tight_layout()

    def variability_plot(self, min_v, max_v):
        list_ytrue, list_ypred = self.return_rebuild_block_matrix()
        num_years = len(list_ytrue)

        w = min([ytrue.shape[0] for ytrue in list_ytrue])
        l = min([ytrue.shape[1] for ytrue in list_ytrue])


        updated_list_ytrue = [mtx[:w, :l] for mtx in list_ytrue]
        stacked_ytrue_matrices = np.stack(updated_list_ytrue)
        ytrue_pixel_wise_variance = np.std(stacked_ytrue_matrices, axis=0)

        updated_list_ypred = [mtx[:w, :l] for mtx in list_ypred]
        stacked_ypred_matrices = np.stack(updated_list_ypred)
        ypred_pixel_wise_variance = np.std(stacked_ypred_matrices, axis=0)

        mae_map_mean, mape_map_mean =[], []
        for i in range(num_years):
            mae_map, mape_map = self.image_mae_mape_map(updated_list_ytrue[i], updated_list_ypred[i])
            mae_map_mean.append(mae_map)
            mape_map_mean.append(mape_map)
        
        mae_pixel_wise_mean = np.mean(mae_map_mean, axis=0)
        mape_pixel_wise_mean = np.mean(mape_map_mean, axis=0)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(1, 4, figsize = (24, 8))

        img1 = axs[0].imshow(ytrue_pixel_wise_variance)
        axs[0].set_title('Yield Observation Variability', fontsize = 14)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(img1,  cax=cax)
        img1.set_clim(min_v, max_v)

        img2 = axs[1].imshow(ypred_pixel_wise_variance)
        axs[1].set_title('Yield Prediction Variability', fontsize = 14)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img2, cax=cax)
        img2.set_clim(min_v, max_v)
        axs[1].get_yaxis().set_visible(False)


        img3 = axs[2].imshow(mae_pixel_wise_mean, cmap = 'viridis') 
        axs[2].set_title('MAE Map (t/ha)', fontsize = 14)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img3, cax=cax)
        img3.set_clim(-1, np.max(mae_pixel_wise_mean))
        axs[2].get_yaxis().set_visible(False)

        img4 = axs[3].imshow(mape_pixel_wise_mean, cmap = 'viridis') #
        axs[3].set_title('MAPE Map (%)', fontsize = 14)
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar3 =fig.colorbar(img4, cax=cax)
        img4.set_clim(-5, 20)
        axs[3].get_yaxis().set_visible(False)

    def _find_closest_value(self, my_list, target):
        """
        Find the value in my_list that is closest to the target value.
        Args:
        my_list (list): A list of numbers.
        target (float or int): The target value to compare against.

        Returns:
        closest_value: The value from my_list that is closest to the target.
        """
        closest_value = my_list[0]
        minimum_diff = abs(my_list[0] - target)

        for value in my_list:
            diff = abs(value - target)
            if diff < minimum_diff:
                closest_value = value
                minimum_diff = diff

        return closest_value

    def segmentation_plot(self, min_v = None, max_v= None):

        ytrue, ypred = self.return_rebuild_block_matrix()
        ytrue_seg, ypred_seg = self._return_seg_matrix(ytrue, ypred)

        plt.rcParams["axes.grid"] = False
        fig, axs = plt.subplots(1, 4, figsize = (24, 8))

        img1 = axs[0].imshow(ytrue)
        axs[0].set_title('Yield Observation', fontsize = 14)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig.colorbar(img1,  cax=cax)
        img1.set_clim(min_v, max_v)

        img2 = axs[1].imshow(ypred)
        axs[1].set_title('Yield Prediction', fontsize = 14)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar2 =fig.colorbar(img2, cax=cax)
        img2.set_clim(min_v, max_v)
        axs[1].get_yaxis().set_visible(False)

        


        class_labels = ['0', '0-17.297', '17.297-22.239', '22.239-32.123', '32.123-49.42', '>49.42']
        cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple'])
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)


        axs[2].imshow(ytrue_seg, cmap=cmap, norm = norm) 
        axs[2].set_title('Yield Zone True', fontsize = 14)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(0.5, len(bounds)), ax = axs[2])
        # cbar.set_ticklabels(class_labels)
        # cbar.set_label('Class')

        axs[3].imshow(ypred_seg, cmap=cmap, norm = norm)
        axs[3].set_title('Yield Zone Pred', fontsize = 14)
        cbar3 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(0.5, len(bounds)), ax = axs[3])
        # cbar3.set_ticklabels(class_labels)
        # cbar3.set_label('Class')

    def _return_seg_matrix(self, ytrue, ypred):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented_ytrue = np.zeros_like(ytrue)
        segmented_ypred = np.zeros_like(ypred)
        # Class 1: Pixel values <= 7
        A = HECTARE_TO_ACRE_SCALE*7
        B = HECTARE_TO_ACRE_SCALE*9
        C = HECTARE_TO_ACRE_SCALE*13
        D = HECTARE_TO_ACRE_SCALE*20

        class_boundaries = [0, 17.297, 22.239, 32.123, 49.42, 75]

        for i, boundary in enumerate(class_boundaries):
            if i == 0:
                continue  # Skip the first boundary as it is the lower limit for class 1
            segmented_ytrue[ytrue >= class_boundaries[i-1]] = i
            segmented_ypred[ypred >= class_boundaries[i-1]] = i

        # segmented_ytrue[ytrue < A] = 1
        # segmented_ypred[ypred < A] = 1

        # segmented_ytrue[ytrue < A] = 1
        # segmented_ypred[ypred < A] = 1
        # # Class 2: Pixel values > 7 and < 9
        # segmented_ytrue[(ytrue >= A) & (ytrue < B)] = 2
        # segmented_ypred[(ypred >= A) & (ypred < B)] = 2
        # # Class 2: Pixel values > 9 and < 13
        # segmented_ytrue[(ytrue >= B) & (ytrue < C)] = 3
        # segmented_ypred[(ypred >= B) & (ypred < C)] = 3
        #     # Class 2: Pixel values > 13 and < 20
        # segmented_ytrue[(ytrue >= C) & (ytrue < D)] = 4
        # segmented_ypred[(ypred >= C) & (ypred < D)] = 4
        # # Class 3: Pixel values >= 20
        # segmented_ytrue[ytrue >= D] = 5
        # segmented_ypred[ypred >= D] = 5

        return segmented_ytrue, segmented_ypred
    
    def _return_full_list_names(self, block_name):

        years = ['2016', '2017', '2018', '2019']

        timeseries_block_names = [int(str(block_name) + year) for year in years]
        updated_timeseries_block_names = [year for year in timeseries_block_names if year in self.test_df['block'].unique()]
        updated_years = [str(year)[-4:] for year in updated_timeseries_block_names]


        if len(str(block_name)) == 1:
            timeseries_block_fullnames = ['LIV_00' + str(block_name) + '_' + year for year in updated_years]
        elif len(str(block_name)) == 2:
            timeseries_block_fullnames = ['LIV_0' + str(block_name) + '_' + year for year in updated_years]
        elif len(str(block_name)) == 3:
            timeseries_block_fullnames = ['LIV_' + str(block_name) + '_' + year for year in updated_years]

        return updated_timeseries_block_names, timeseries_block_fullnames

    def return_rebuild_block_matrix(self, block_name, year: int):
        if year == 2016:
            year_id = 0
        elif year == 2017:
            year_id = 1
        elif year == 2018:
            year_id = 2
        elif year == 2019:
            year_id = 3


        timeseries_block_names, timeseries_block_fullnames = self._return_full_list_names(block_name)
        timeseries_block_names, timeseries_block_fullnames = timeseries_block_names[year_id], timeseries_block_fullnames[year_id]

        blocks_df = self.test_df.groupby(by = 'block')
        
        # for idx, block in enumerate(timeseries_block_names):

        this_block_df = blocks_df.get_group(timeseries_block_names)
        # print(f"{block}: {this_block_df.shape}")

        res = {key: configs.blocks_size[key] for key in configs.blocks_size.keys() & {timeseries_block_fullnames}}
        list_d = res.get(timeseries_block_fullnames)
        block_x_size = int(list_d[0]/10.0)
        block_y_size = int(list_d[1]/10.0)

        true_out = np.full((block_x_size, block_y_size), -1)  
        pred_out_w1 = np.full((block_x_size, block_y_size), -1)
        pred_out_w2 = np.full((block_x_size, block_y_size), -1)
        pred_out_w3 = np.full((block_x_size, block_y_size), -1)
        pred_out_w4 = np.full((block_x_size, block_y_size), -1)
        pred_out_w5 = np.full((block_x_size, block_y_size), -1) 
        pred_out_w6 = np.full((block_x_size, block_y_size), -1)
        pred_out_w7 = np.full((block_x_size, block_y_size), -1)
        pred_out_w8 = np.full((block_x_size, block_y_size), -1)
        pred_out_w9 = np.full((block_x_size, block_y_size), -1)
        pred_out_w10 = np.full((block_x_size, block_y_size), -1) 
        pred_out_w11 = np.full((block_x_size, block_y_size), -1)
        pred_out_w12 = np.full((block_x_size, block_y_size), -1)
        pred_out_w13 = np.full((block_x_size, block_y_size), -1)
        pred_out_w14 = np.full((block_x_size, block_y_size), -1)
        pred_out_w15 = np.full((block_x_size, block_y_size), -1) 

        for x in range(block_x_size):
            for y in range(block_y_size):
                new = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)]
                if len(new) > 0:
                    true_out[x, y] = new['ytrue']* HECTARE_TO_ACRE_SCALE
                    pred_out_w1[x, y] = new['ypred_w1']* HECTARE_TO_ACRE_SCALE
                    pred_out_w2[x, y] = new['ypred_w2']* HECTARE_TO_ACRE_SCALE
                    pred_out_w3[x, y] = new['ypred_w3']* HECTARE_TO_ACRE_SCALE
                    pred_out_w4[x, y] = new['ypred_w4']* HECTARE_TO_ACRE_SCALE
                    pred_out_w5[x, y] = new['ypred_w5']* HECTARE_TO_ACRE_SCALE
                    pred_out_w6[x, y] = new['ypred_w6']* HECTARE_TO_ACRE_SCALE
                    pred_out_w7[x, y] = new['ypred_w7']* HECTARE_TO_ACRE_SCALE
                    pred_out_w8[x, y] = new['ypred_w8']* HECTARE_TO_ACRE_SCALE
                    pred_out_w9[x, y] = new['ypred_w9']* HECTARE_TO_ACRE_SCALE
                    pred_out_w10[x, y] = new['ypred_w10']* HECTARE_TO_ACRE_SCALE
                    pred_out_w11[x, y] = new['ypred_w11']* HECTARE_TO_ACRE_SCALE
                    pred_out_w12[x, y] = new['ypred_w12']* HECTARE_TO_ACRE_SCALE
                    pred_out_w13[x, y] = new['ypred_w13']* HECTARE_TO_ACRE_SCALE
                    pred_out_w14[x, y] = new['ypred_w14']* HECTARE_TO_ACRE_SCALE
                    pred_out_w15[x, y] = new['ypred_w15']* HECTARE_TO_ACRE_SCALE

                        # if mode == 'mean':
                        #     pred_out[x, y] = new['ypred_w15'].mean()* HECTARE_TO_ACRE_SCALE
                        # elif mode == 'close': 
                        #     pred_out[x, y] = self._find_closest_value(list(new['ypred_w15']), new['ytrue'].mean())* HECTARE_TO_ACRE_SCALE
                        # elif mode == 'median': 
                        #     pred_out[x, y] = new['ypred_w15'].quantile(0.5)* HECTARE_TO_ACRE_SCALE

        list_ypred = [pred_out_w1, pred_out_w2, pred_out_w3, pred_out_w4, pred_out_w5,
                            pred_out_w6, pred_out_w7, pred_out_w8, pred_out_w9, pred_out_w10,
                            pred_out_w11, pred_out_w12, pred_out_w13, pred_out_w14, pred_out_w15]
        # Check if any pixel in true_out is -1 and update pred_out_w{i} accordingly
        # for x in range(block_x_size):
        #     for y in range(block_y_size):
        #         if  true_out[x, y] != -1: 
        #             for pred_matrix in list_ypred:
        #                 # if true_out[x, y]  -  pred_matrix[x, y] == 5:
        #                 pred_matrix[x, y] = pred_matrix[x, y] + 5


        return true_out, list_ypred     

    def image_mae_mape_map(self, ytrue, ypred): 

        w = ytrue.shape[0]
        h = ytrue.shape[1] 

        ytrue_flat = ytrue.ravel()
        ypred_flat = ypred.ravel()
        out_mae    = np.empty_like(ytrue_flat)
        out_mape   = np.empty_like(ytrue_flat)

        for i in range(len(ytrue_flat)):
            if (ytrue_flat[i] == -1) and (ypred_flat[i] == -1):
                out_mae[i]  = -10
                out_mape[i] = -10
            # elif abs(ytrue_flat[i] - ypred_flat[i]) > 11:
            #     out_mae[i]  = -10
            #     out_mape[i] = -10
            else: 
                out_mae[i]  = abs(ytrue_flat[i] - ypred_flat[i])
                out_mape[i] = ((abs(ytrue_flat[i] - ypred_flat[i]))/ytrue_flat[i])*100

        out1 = out_mae.reshape(ytrue.shape[0], ytrue.shape[1])
        out2 = out_mape.reshape(ytrue.shape[0], ytrue.shape[1])

        return out1, out2



LOW_EXTREME_RANGE_BLOCKS = [32016, 32017, 32018, 32019, 382016, 1032016, 1072016]
COMMON_RANGE_BLOCKS = [72016, 72019, 122016, 122017, 122018, 122019, 142016, 142017, 142018, 142019, 162016, 162017, 162018, 172016, 172017, 172018, 182016, 182017, 182018, 182019, 382017, 382018, 682016, 682017, 762016, 
                       762017, 762018, 762019, 1022016, 1022017, 1022018, 1022019, 1032017, 1032018, 1032019, 1072017, 1072018, 1072019, 1112016, 1112017, 1112018, 1112019, 1762017, 1762018, 1762019]
HIGH_EXTREME_RANGE_BLOCKS = [72017, 72018, 162019]

class TextScoresAnalysis():
    def __init__(self, exp_name: str, batch_size: int, layer: int, head:int, lrp: str = False):
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.head = head
        self.layer = layer

        self.root_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep'
        self.attn_dir = os.path.join(self.root_dir , 'EXP_' + self.exp_name + '/attn_scores')

        self.df = pd.read_csv(os.path.join(self.root_dir, 'EXP_' + self.exp_name, self.exp_name + '_train.csv'))
        self.TextEncoder = AutoTokenizer.from_pretrained("distilbert-base-uncased")#tiktoken.get_encoding('gpt2')

    def extreme_categ_visualize_text(self):

        main_dict = self._all_blcok_scores()

        low_extreme_dict = {key: main_dict[key] for key in LOW_EXTREME_RANGE_BLOCKS if key in main_dict}
        common_dict = {key: main_dict[key] for key in COMMON_RANGE_BLOCKS if key in main_dict}
        high_extreme_dict = {key: main_dict[key] for key in HIGH_EXTREME_RANGE_BLOCKS if key in main_dict}

        low_mean = self._calculate_mean_for_dict(low_extreme_dict)[-1, :, :, self.layer]
        low_mean = low_mean[2:, 2:]
        low_mean = np.mean(low_mean, axis=0)
        low_mean = self._normalize_array(low_mean)

        common_mean = self._calculate_mean_for_dict(common_dict)[-1, :, :, self.layer]
        common_mean = common_mean[2:, 2:]
        common_mean = np.mean(common_mean, axis=0)
        common_mean = self._normalize_array(common_mean)

        high_mean = self._calculate_mean_for_dict(high_extreme_dict)[-1, :, :, self.layer]
        high_mean = high_mean[2:, 2:]
        high_mean = np.mean(high_mean, axis=0)
        high_mean = self._normalize_array(high_mean)

        words = self._text_decode(block = 162017)
        self._visualize_text(words, low_mean)
        self._visualize_text(words, common_mean)
        self._visualize_text(words, high_mean)

    def single_visualize_text(self, block: None):
        """Use decoded text and attention scores to create a visualization."""
        words, attention_mask = self._text_decode(block = block)
        importances = self._single_calc_attn(block = block)
        print(len(words), len(attention_mask), len(importances))

        filtered_words = [word for word, mask in zip(words, attention_mask) if mask]
        filtered_importances = [importance for importance, mask in zip(importances, attention_mask) if mask]


        self._visualize_text(filtered_words, filtered_importances)
    
    def _calculate_mean_for_dict(self, dictionary):

        all_values = []
        for key, values in dictionary.items():
            all_values.append(values)
        
        # Assuming all values are numeric matrices of the same shape
        if all_values:
            mean_value = np.mean(all_values, axis=0)
        else:
            mean_value = None  # or handle empty case as needed
        
        return mean_value
   
    def _find_file_in_directory(self, block):
        """
        Search for a specific .txt file in a given directory and return its full path.
        
        :param directory: Directory to search in
        :param filename: Name of the file to search for
        :return: Full path of the file if found, else None
        """

        directory = '/data2/hkaman/Data/Livingston/text'
        # Ensure the filename ends with .txt
        block = str(block)
        if not block.endswith('.txt'):
            block_root_name = block[:-4]

            if len(str(block_root_name)) == 1:
                block_fullnames = 'LIV_00' + str(block_root_name) + '.txt'
            elif len(str(block_root_name)) == 2:
                block_fullnames = 'LIV_0' + str(block_root_name) + '.txt'
            elif len(str(block_root_name)) == 3:
                block_fullnames = 'LIV_' + str(block_root_name) + '.txt'

        
        # Search for the file in the directory
        for root, _, files in os.walk(directory):
            if block_fullnames in files:
                return os.path.join(root, block_fullnames)
        
        return None

    def _return_text_of_block(self, block):
        # Extract the file extension to determine how to process it
        # block_root_name = block_name[:-4] + 
        text_path = self._find_file_in_directory(block = block)

        _, file_extension = os.path.splitext(text_path)
        
        if file_extension.lower() == '.pdf':
            # Handle PDF files
            pdf_loader = PdfReader(open(text_path, "rb"))
            file_text = ""
            for page_num in range(len(pdf_loader.pages)):
                pdf_page = pdf_loader.pages[page_num]
                if pdf_page.extract_text() is not None:
                    file_text += pdf_page.extract_text()
            return file_text
        
        elif file_extension.lower() == '.txt':
            # Handle text files
            with open(text_path, "r", encoding="utf-8") as file:
                file_text = file.read()
            return file_text
        
        else:
            # Unsupported file type
            raise ValueError("Unsupported file format: " + file_extension)

    def encode(self, text):
        # Encode the text into tokens
        return self.TextEncoder(text, return_tensors="pt")['input_ids'][0].tolist()
    
    def _text_encode(self, block):
        max_length = 250
        text = self._return_text_of_block(block = block)
        tokens = self.TextEncoder(text, return_tensors="pt", padding='max_length', max_length=max_length)['input_ids'][0].tolist()
        attention_mask =self.TextEncoder(text, return_tensors="pt", padding='max_length', max_length=max_length)['attention_mask'][0].bool()
        # inputs = self.TextEncoder(text, padding='max_length', truncation=True, return_tensors="pt", max_length=max_length)
        # inputs = {key: value for key, value in inputs.items()}
        
        # encoded_texts = [self.TextEncoder.encode(text) for text in text]
        # max_length = 249 
        # padded_texts = [text[:max_length] + [0] * (max_length - len(text)) for text in encoded_texts]
        # print(padded_texts)
        # tokens = np.array(padded_texts, dtype = np.uint32)#.to(self.device)
        # arr = np.load('/home/hkaman/Documents/multimodel-transformers-vye/Junky/attn_scores.npy', allow_pickle=True).item()
        # tokens = arr[10]['tokens']

        return tokens, attention_mask
    
    def decode_single_token_bytes(self, token_id):
        # Convert a single token ID to its corresponding text token
        token = self.TextEncoder.convert_ids_to_tokens(token_id)
        # Decode the token to a readable string
        return self.TextEncoder.decode([token_id])

    def _text_decode(self, block):
        """Decode tokens using the TextEncoder."""
        # Decode both the text and get token offsets
        tokens, attention_mask = self._text_encode(block = block)
        # attention_mask = tokens['attention_mask'].bool()
        words = [self.decode_single_token_bytes(token) for token in tokens]
        # words = [t.decode('utf-8') for t in token_bytes]
        return words, attention_mask
    
    def _single_calc_attn(self, block):
        """Calculate the average attention score for a specific layer and head, excluding the first token."""
        # Extract the attention scores for the specified head and layer
        # if ana_status == 'single': 
        attn_scores = self._single_blcok_scores(block = block)[self.head, :, :, self.layer]#self.attns_arr[self.head, :, :, self.layer]
        attn_scores = attn_scores[2:, 2:]
        avg_attn_scores = np.mean(attn_scores, axis=0)
        norm_attn_scores = self._normalize_array(avg_attn_scores)
        return norm_attn_scores
        
    def _global_cal_attn(self):
        # elif ana_status == 'global':
        scores = self._all_blcok_scores()
        return scores
    
    def _normalize_array(self, values):
        min_old = values.min()
        max_old = values.max()
        min_new, max_new = -1, 1
        normalized_values = [(value - min_old) / (max_old - min_old) * (max_new - min_new) + min_new for value in values]
        return np.array(normalized_values, dtype= np.float32)
    
    def _format_special_tokens(self, word):
        # Strip underscores often used in tokenized outputs
        return word.replace('_', ' ')

    def _get_color(self, attr):
        # clip values to prevent CSS errors (Values should be from [-1,1])
        attr = max(-1, min(1, attr))
        if attr > 0:
            hue = 120
            sat = 75
            lig = 100 - int(50 * attr)
        else:
            hue = 0
            sat = 75
            lig = 100 - int(-40 * attr)
        return "hsl({}, {}%, {}%)".format(hue, sat, lig)

    def _format_word_importances(self, words, importances):
        tags = ["<td>"]
        for word, importance in zip(words, importances):
            color = self._get_color(importance)
            tags.append(
                '<mark style="background-color: {color}; opacity:1.0; line-height:1.75">'
                '<font color="black"> {word} </font></mark>'.format(color=color, word=word)
            )
        tags.append("</td>")
        return "".join(tags)

    def _visualize_text(self, words, importances, legend=True):
        print(len(words), len(importances))
        assert len(words) == len(importances), "Words and importances must have the same length."
        
        dom = ["<table style='width: 100%;'>"]
        dom.append(
            "<tr>{}</tr>".format(self._format_word_importances(words, importances))
        )
        
        if legend:
            dom.append(
                '<div style="border-top: 1px solid; margin-top: 5px; padding-top: 5px; display: inline-block">'
            )
            dom.append("<b>Legend: </b>")
            for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
                dom.append(
                    '<span style="display: inline-block; width: 20px; height: 10px; border: 1px solid; background-color: {value};"></span> {label}  '.format(
                        value=self._get_color(value), label=label
                    )
                )
            dom.append("</div>")
        
        dom.append("</table>")
        html = HTML("".join(dom))
        display(html)
        return html
    
    def _return_blocks_patch_info(self):
        unique_blocks = pd.unique(self.df['block'])
        block_patch_range = {}

        lower_ = 0
        for idx, block in enumerate(unique_blocks):
            size = len(self.df[self.df['block'] == block]) / 256
            block_patch_range[block] = (lower_, size + lower_)
            lower_ += size

        return block_patch_range
    
    def _single_blcok_scores(self, block):


        range_dict = self._return_blocks_patch_info()

        key = block
        lower_bound = int(range_dict[key][0])
        upper_bound = int(range_dict[key][1])

        # Calculate the starting and ending file indices
        start_file_index = lower_bound // self.batch_size
        end_file_index = (upper_bound - 1) // self.batch_size  # Subtract 1 to handle inclusive upper bound correctly

        # List to store slices of arrays for averaging
        slices = []

        # Loop over the necessary file indices
        for file_index in range(start_file_index, end_file_index + 1):
            filename = f"train_attn_scores_{file_index}.npy"
            file_path = os.path.join(self.attn_dir, filename)
            
            if os.path.exists(file_path):
                array = np.load(file_path)

                # Calculate slice bounds within the current array
                slice_start = lower_bound - self.batch_size * file_index
                slice_end = upper_bound - self.batch_size * file_index

                # Adjust slice bounds to fit within the current array
                slice_start = max(0, slice_start)
                slice_end = min(self.batch_size, slice_end)

                if slice_start < slice_end:  # Ensure there is something to slice
                    slices.append(array[slice_start:slice_end])

        # Concatenate all slices along the first axis and compute the mean
        if slices:
            combined_array = np.concatenate(slices, axis=0)
            mean_array = np.percentile(combined_array, 95, axis=0)#np.mean(combined_array, axis=0)

        return mean_array
        
    def _all_blcok_scores(self):
        scores = {}

        range_dict = self._return_blocks_patch_info()
        # Iterate over each specified range
        for key, bounds in range_dict.items():
            lower_bound = int(bounds[0])
            upper_bound = int(bounds[1])

            # Calculate the starting and ending file indices
            start_file_index = lower_bound // self.batch_size
            end_file_index = (upper_bound - 1) // self.batch_size  # Subtract 1 to handle inclusive upper bound correctly

            # List to store slices of arrays for averaging
            slices = []

            # Loop over the necessary file indices
            for file_index in range(start_file_index, end_file_index + 1):
                filename = f"train_attn_scores_{file_index}.npy"
                file_path = os.path.join(self.attn_dir, filename)
                
                if os.path.exists(file_path):
                    array = np.load(file_path)

                    # Calculate slice bounds within the current array
                    slice_start = lower_bound - self.batch_size * file_index
                    slice_end = upper_bound - self.batch_size * file_index

                    # Adjust slice bounds to fit within the current array
                    slice_start = max(0, slice_start)
                    slice_end = min(self.batch_size, slice_end)

                    if slice_start < slice_end:  # Ensure there is something to slice
                        slices.append(array[slice_start:slice_end])

            # Concatenate all slices along the first axis and compute the mean
            if slices:
                combined_array = np.concatenate(slices, axis=0)
                mean_array = np.mean(combined_array, axis=0) #np.percentile(combined_array, 95, axis=0)
                scores[key] = mean_array

        return scores


#=========================================================================#
#================================ Predict ================================#
#=========================================================================#
import torch
import numpy as np
from models import configs
from models.configs import Configs, set_seed
set_seed(1987)

from models.cvt import ClimMgmtAware_ViT
# from models.ClimMgmtAwareViTLRP import ClimMgmtAware_ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

config = Configs(
    img_size = 16, 
    patch_size = 8, 
    embed_dim = 768, 
    context_dim = 768,
    mlp_dim = 512, 
    pool = 'cls',
    in_channels = 8,
    out_channels = 1, 
    num_heads = 8, 
    num_layers = 6, 
    cond = False,
    multi_conv = False,
    attn_dropout = .3, 
    proj_dropout = .3, 
    drop_path = 0.0,
    post_norm = False, 
    vis = True, 
    tokenizer = 'EC',
    mask_modality = None
    ).call()


def predict(data_loader, exp_name: str, keyword: str, new_value: float):

    exp_output_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep/' + 'EXP_' + exp_name
    best_model_name = os.path.join(exp_output_dir, 'best_model_' + exp_name + '.pth')

    model = ClimMgmtAware_ViT(config).to(device)
    model.load_state_dict(torch.load(best_model_name))


    output_files =[]

    with torch.no_grad():
        for batch, sample in enumerate(data_loader):
            x = sample['image'].to(device)
            met = sample['met'].to(device)
            y = sample['mask'].detach().cpu().numpy()
            block_id = sample['block']
            block_cultivar_id = sample['cultivar']
            block_x_coords = sample['X']
            block_y_coords = sample['Y']
            embmatrix = sample['EmbText']

            pred_list, _ = model(img = x, 
                                    context = embmatrix, 
                                    met = met, 
                                    yz = None, 
                                    cond = False) 

            this_batch = {"block": block_id, 
                                "cultivar": block_cultivar_id, 
                                "X": block_x_coords, "Y": block_y_coords,
                                "ytrue": y}

            # Dynamically add predictions to the dictionary
            for i, pred in enumerate(pred_list):
                key = f"ypred_w{i+1}"  # Creates keys like ypred_w1, ypred_w2, ..., ypred_wN
                this_batch[key] = pred.detach().cpu().numpy()

            output_files.append(this_batch)

        modified_df = _return_modified_pred_df(output_files, None, 16)
        cleaned_keyword = re.sub(r'[^a-zA-Z0-9_]', '', keyword)
        analysis_output_dir = os.path.join(exp_output_dir, 'sensivity')
        isExist  = os.path.isdir(analysis_output_dir)

        if not isExist:
            os.makedirs(analysis_output_dir)

        test_df_name = os.path.join(analysis_output_dir, exp_name + '_test_'+ cleaned_keyword + '_' + str(new_value) + '.csv')
        modified_df.to_csv(test_df_name)

        print("Done!")

def _return_modified_pred_df(pred_npy, blocks_list, wsize=None):
    if blocks_list is None: 
        all_block_names = [dict['block'] for dict in pred_npy]#[0]
        blocks_list = list(set(item for sublist in all_block_names for item in sublist))


    OutDF = pd.DataFrame()
    columns = ['block', 'cultivar', 'x', 'y', 'ytrue']
    data = {col: [] for col in columns}  # Initialize dictionary for DataFrame

    # Initialize lists for predictions dynamically based on the first item's keys
    pred_keys = [key for key in pred_npy[0].keys() if key.startswith('ypred')]
    for key in pred_keys:
        data[key] = []

    for block in blocks_list:
        name_split = os.path.split(block)[-1]
        block_name = name_split.replace(name_split[7:], '')
        root_name = name_split.replace(name_split[:4], '').replace(name_split[3], '')
        block_id = root_name
        
        res = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {block_name}}
        list_d = res.get(block_name)
        cultivar_id = list_d[1]
    
        for l in range(len(pred_npy)):
            tb_pred_indices = [i for i, x in enumerate(pred_npy[l]['block']) if x == block]
            if len(tb_pred_indices) !=0:   
                for index in tb_pred_indices:

                    x0 = pred_npy[l]['X'][index]
                    y0 = pred_npy[l]['Y'][index]
                    x_vector, y_vector = _xy_vector_generator(x0, y0, wsize)
                    data['x'].append(x_vector)
                    data['y'].append(y_vector)
                    data['ytrue'].append(pred_npy[l]['ytrue'][index].flatten())

                    tb_block_id = np.array(len(pred_npy[l]['ytrue'][index].flatten())*[block_id], dtype=np.int32)
                    data['block'].append(tb_block_id)

                    tb_cultivar_id = np.array(len(pred_npy[l]['ytrue'][index].flatten())*[cultivar_id], dtype=np.int8)
                    data['cultivar'].append(tb_cultivar_id)



                    # Handle predictions dynamically
                    for key in pred_keys:
                        flattened_pred = pred_npy[l][key][index].flatten()
                        data[key].append(flattened_pred)

    empty_dict = {key: None for key in data.keys()}
    # Convert lists to numpy arrays for consistency
    for key in data:
        if data[key]:  # Ensure there's data to concatenate
            # print(len(data[key]))
            output = np.concatenate(data[key])
            empty_dict[key] = output
            # print(key, output.shape)

    # Create DataFrame from data dictionary
    OutDF = pd.DataFrame(empty_dict)
    return OutDF

def _xy_vector_generator(x0, y0, wsize):

    x_vector, y_vector = [], []
    
    for i in range(x0, x0+wsize):
        for j in range(y0, y0+wsize):
            x_vector.append(i)
            y_vector.append(j)

    return x_vector, y_vector 

def _load_text_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        # Handle PDF files
        pdf_loader = PdfReader(open(file_path, "rb"))
        file_text = ""
        for page_num in range(len(pdf_loader.pages)):
            pdf_page = pdf_loader.pages[page_num]
            if pdf_page.extract_text() is not None:
                file_text += pdf_page.extract_text()
        return file_text
    
    elif file_extension.lower() == '.txt':
        # Handle text files
        with open(file_path, "r", encoding="utf-8") as file:
            file_text = file.read()
        return file_text
    
    else:
        raise ValueError("Unsupported file format: " + file_extension)

def extract_keyword_values(text, keywords):
    values = {}
    for keyword in keywords:
        if keyword == 'electrical conductivity':
            # Specific pattern for electrical conductivity, handling "at" and potential line breaks
            pattern = rf"{re.escape(keyword)}.*?at\s*([\d\.]+)" #pattern = rf"{re.escape(keyword)}.*?at\s*([\d\.]+)"
        else:
            # General pattern for other keywords
            pattern = rf"\b{re.escape(keyword)}\b.*?([\d\.]+)\s*(?:[a-zA-Z%\/\(\)]*)"
        

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL) #re.search(pattern, text, re.IGNORECASE)
        
        # print(f'{keyword}, match: {match}')  # Debugging print statement
        
        if match:
            try:
                # Try to extract and convert the matched value to a float
                value = float(match.group(1))  # This captures the number part
                # print(f'values: {value}')  # Debugging print statement
                if keyword not in values:
                    values[keyword] = []
                values[keyword].append(value)
            except ValueError:
                continue  # Ignore if conversion fails
    return values

def process_texts(dataloader, keywords):
    all_values = {keyword: [] for keyword in keywords}
    
    for sample in dataloader:
        # text_path = 
        text = sample['EmbText'][0] #_load_text_file(text_path)
        # Only process texts longer than 1000 characters
        if len(text) > 100:
            keyword_values = extract_keyword_values(text, keywords)
            for keyword, values in keyword_values.items():
                all_values[keyword].extend(values)
    
    # Compute 10th, 50th, and 90th percentiles for each keyword
    percentiles = {}
    for keyword, values in all_values.items():
        if values:
            percentiles[keyword] = {
                '10th': np.percentile(values, 10),
                '50th': np.percentile(values, 50),
                '90th': np.percentile(values, 90)
            }
        else:
            percentiles[keyword] = {'10th': None, '50th': None, '90th': None}
    
    return percentiles





def plot_sensitivity_analysis(exp_name: str, 
                              keyword: str, values: list):
    exp_output_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep/' + 'EXP_' + exp_name
    analysis_output_dir = os.path.join(exp_output_dir, 'sensivity')
    
    # Load the original dataframe
    original_df = pd.read_csv(os.path.join(exp_output_dir, exp_name + '_test.csv'))
    
    # Load the three sensitivity dataframes for the keyword
    cleaned_keyword = re.sub(r'[^a-zA-Z0-9_]', '', keyword)

    sens_df_10th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values[0])}.csv"))
    sens_df_50th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values[1])}.csv"))
    sens_df_90th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values[2])}.csv"))
    
    # Create a list of dataframes for iteration
    dfs = {
        "Original": original_df,
        "10th Percentile": sens_df_10th,
        "50th Percentile": sens_df_50th,
        "90th Percentile": sens_df_90th
    }
    
    # Define the bin edges and labels
    bin_edges = np.arange(0, 71, 7)
    bin_labels = [f"{i}-{i+7}" for i in bin_edges[:-1]]
    
    # Create a figure for the boxplot
    plt.figure(figsize=(18, 6))  # Wider figure to match horizontal layout
    
    # Colors for each boxplot
    box_colors = ['gray', 'lightblue', 'lightgreen', 'lightcoral']
    
    # Iterate through bins
    for i in range(len(bin_edges) - 1):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        
        # Collect ypred_w1 values for all dataframes within the bin range based on ytrue
        ypred_bin_values = []
        for label, df in dfs.items():
            bin_data = df[(df['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (df['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1']*HECTARE_TO_ACRE_SCALE
            ypred_bin_values.append(bin_data)
        
        # Plot vertically and shift the positions of boxplots horizontally for each bin
        x_positions = np.array([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3])  # Shifting x-axis for vertical boxplots
        for j, (ypred, x_pos, color) in enumerate(zip(ypred_bin_values, x_positions, box_colors)):
            boxplot = plt.boxplot(ypred, vert=True, positions=[x_pos], widths=0.6, patch_artist=True, showfliers=False)
            
            # Set box color
            for patch in boxplot['boxes']:
                patch.set_facecolor(color)

        if i < len(bin_edges) - 2:
            plt.axvline(x=(i + 1) * 4 - 0.5, color='gray', linestyle='--', linewidth=1)
    
    
    # Set y-axis limits and labels (Yield)
    plt.ylim([0, 70])
    plt.ylabel("Yield (0-70)")
    
    # Set x-axis labels and ticks (Bins)
    x_tick_positions = np.arange(2, len(bin_labels) * 4, 4)  # Adjusted for 10 bins
    plt.xticks(x_tick_positions, bin_labels)  # Ensure 10 tick positions match 10 labels
    plt.xlabel("Bins")
    
    # Add legend
    plt.legend([plt.Line2D([0], [0], color=c, lw=4) for c in box_colors], 
               ['Original', '10th Percentile', '50th Percentile', '90th Percentile'], loc='upper left')
    
    # Add title and adjust layout
    plt.title(f"Sensitivity Analysis for {keyword}")
    plt.tight_layout()
    
    # Show the plot
    plt.show()



# def plot_mean_difference(exp_name: str, keywords_dict: dict):
#     exp_output_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep/' + 'EXP_' + exp_name
#     analysis_output_dir = os.path.join(exp_output_dir, 'sensivity')

#     # Define the bin edges
#     bin_edges = np.arange(0, 71, 7)
    
#     # Create a figure with 1 row and 10 columns (subplots for each bin)
#     fig, axes = plt.subplots(1, len(bin_edges) - 1, figsize=(20, 5), sharey=True)
    
#     # Iterate through bins and create subplots
#     for i, ax in enumerate(axes):
#         bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        
#         y_offset = len(keywords_dict)  # For offsetting the y-axis for each keyword
        
#         # For each keyword, calculate the means and their differences
#         for idx, (keyword, value) in enumerate(keywords_dict.items()):
#             values_list = list(value.values())
#             original_df = pd.read_csv(os.path.join(exp_output_dir, exp_name + '_test.csv'))
#             cleaned_keyword = re.sub(r'[^a-zA-Z0-9_]', '', keyword)
#             sens_df_10th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values_list[0])}.csv"))
#             sens_df_50th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values_list[1])}.csv"))
#             sens_df_90th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values_list[2])}.csv"))
            


#             # Calculate means for the current keyword
#             original_mean = original_df[(original_df['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (original_df['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE
#             mean_10th = sens_df_10th[(sens_df_10th['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (sens_df_10th['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE
#             mean_50th = sens_df_50th[(sens_df_50th['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (sens_df_50th['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE
#             mean_90th = sens_df_90th[(sens_df_90th['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (sens_df_90th['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE

#             # Calculate the differences from the original mean
#             diff_10th = mean_10th - original_mean
#             diff_50th = mean_50th - original_mean
#             diff_90th = mean_90th - original_mean

#             # Plot the differences as dot plots
#             # ax.plot([diff_10th], [y_offset - idx], 'o', color='blue', label='10th Percentile' if i == 0 and idx == 0 else "")  # Dot for 10th percentile
#             # ax.plot([diff_50th], [y_offset - idx], 'o', color='green', label='50th Percentile' if i == 0 and idx == 0 else "")  # Dot for 50th percentile
#             # ax.plot([diff_90th], [y_offset - idx], 'o', color='red', label='90th Percentile' if i == 0 and idx == 0 else "")  # Dot for 90th percentile
#             # Modify these lines to use horizontal bar plots with a shift
#             bar_width = 0.2  # Set the width of each bar
#             # Plot bar for 10th Percentile with a horizontal shift
#             ax.barh(y=[y_offset - idx + bar_width], width=diff_10th, height=bar_width, color='blue', label='10th Percentile' if i == 0 and idx == 0 else "")
#             # Plot bar for 50th Percentile with a horizontal shift
#             ax.barh(y=[y_offset - idx], width=diff_50th, height=bar_width, color='green', label='50th Percentile' if i == 0 and idx == 0 else "")
#             # Plot bar for 90th Percentile with a horizontal shift
#             ax.barh(y=[y_offset - idx - bar_width], width=diff_90th, height=bar_width, color='red', label='90th Percentile' if i == 0 and idx == 0 else "")
#             # Add a horizontal dashed line separating keywords
#             ax.axhline(y=y_offset - idx - 0.5, color='gray', linestyle='--', linewidth=1)
        
#         # Set x-axis range from -1 to 1 (centered around original mean)
#         ax.set_xlim([-1, 1])
        
#         # Set title for the bin
#         ax.set_title(f"Bin {bin_start}-{bin_end}")
    
#     # Set common labels and legend
#     fig.suptitle(f"Mean Differences from Original for Keywords")
#     fig.text(0.5, 0.04, "Difference from Original Mean", ha="center")
#     fig.text(0.04, 0.5, "Keywords", va="center", rotation="vertical")
    
#     # Add legend
#     plt.legend(loc="upper right")
    
#     # Adjust layout
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
#     # Show the plot
#     plt.show()


def plot_mean_difference(exp_name: str, keywords_dict: dict, block: int):
    exp_output_dir = '/data2/hkaman/Projects/ViT/EXPs/Sep/' + 'EXP_' + exp_name
    analysis_output_dir = os.path.join(exp_output_dir, 'sensivity')

    # Define the bin edges
    bin_edges = np.arange(0, 71, 7)
    
    # Extract keyword names from the dictionary
    keyword_names = list(keywords_dict.keys())
    
    # Create a figure with 1 row and 10 columns (subplots for each bin)
    fig, axes = plt.subplots(1, len(bin_edges) - 1, figsize=(20, 4), sharey=True)
    
    # Iterate through bins and create subplots
    for i, ax in enumerate(axes):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        
        y_offset = len(keyword_names)  # For offsetting the y-axis for each keyword
        
        # For each keyword, calculate the means and their differences
        for idx, (keyword, value) in enumerate(keywords_dict.items()):
            values_list = list(value.values())
            original_df = pd.read_csv(os.path.join(exp_output_dir, exp_name + '_test.csv'))
            cleaned_keyword = re.sub(r'[^a-zA-Z0-9_]', '', keyword)
            sens_df_10th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values_list[0])}.csv"))
            sens_df_50th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values_list[1])}.csv"))
            sens_df_90th = pd.read_csv(os.path.join(analysis_output_dir, exp_name + '_test_'+ f"{cleaned_keyword}_{str(values_list[2])}.csv"))
            if block:
                sens_df_10th = sens_df_10th[sens_df_10th['block'] == block]
                sens_df_50th = sens_df_50th[sens_df_50th['block'] == block]
                sens_df_90th = sens_df_90th[sens_df_90th['block'] == block]

            # Calculate means for the current keyword
            original_mean = original_df[(original_df['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (original_df['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE
            mean_10th = sens_df_10th[(sens_df_10th['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (sens_df_10th['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE
            mean_50th = sens_df_50th[(sens_df_50th['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (sens_df_50th['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE
            mean_90th = sens_df_90th[(sens_df_90th['ytrue']*HECTARE_TO_ACRE_SCALE >= bin_start) & (sens_df_90th['ytrue']*HECTARE_TO_ACRE_SCALE < bin_end)]['ypred_w1'].mean()*HECTARE_TO_ACRE_SCALE

            # Calculate the differences from the original mean
            diff_10th = mean_10th - original_mean
            diff_50th = mean_50th - original_mean
            diff_90th = mean_90th - original_mean

            # Plot the differences as bar plots
            bar_width = 0.1  # Set the width of each bar
            ax.barh(y=[y_offset - idx + bar_width], width=diff_10th, height=bar_width, color='blue', label='10th Percentile' if i == 0 and idx == 0 else "")
            ax.barh(y=[y_offset - idx], width=diff_50th, height=bar_width, color='green', label='50th Percentile' if i == 0 and idx == 0 else "")
            ax.barh(y=[y_offset - idx - bar_width], width=diff_90th, height=bar_width, color='red', label='90th Percentile' if i == 0 and idx == 0 else "")
            
            # Add a horizontal dashed line separating keywords
            ax.axhline(y=y_offset - idx - 0.5, color='gray', linestyle='--', linewidth=1)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # Set x-axis range from -5 to 5 (centered around original mean)
        ax.set_xlim([-1, 1])
        
        # Set title for the bin
        ax.set_title(f"Bin {bin_start}-{bin_end}")
        
        # Set the y-axis ticks to show keyword names instead of values
        ax.set_yticks([y_offset - i for i in range(len(keyword_names))])
        ax.set_yticklabels(keyword_names, rotation=90)
    
    # Set common labels and legend
    fig.suptitle(f"Mean Differences from Original for Keywords")
    
    # Add x-axis label with parentheses legend
    fig.text(0.5, 0.04, "Difference from Original Mean (Blue: 10th, Green: 50th, Red: 90th Percentile)", ha="center")
    # fig.text(0.04, 0.5, "Keywords", va="center", rotation="vertical")
    
    # Add legend
    plt.legend(loc="upper right")
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show the plot
    plt.show()