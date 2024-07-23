import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model.configs import blocks_information, blocks_size


#-------------------------------------------------------------------------------------------#
#                                    Vineyard Data Vis                                      #
#-------------------------------------------------------------------------------------------#
def dual_emp_effective_hist_plot(emp, effective):
     
    fig, axs = plt.subplots(1, 2 , figsize = (15, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 31, 1)
    ax1 = sns.barplot(x = bins_value, y= emp, color = sns.color_palette()[0], width = 0.9, ax = axs[0])
    ax1.set_title("Emprical Label Distribution", fontsize = 14)
    ax2 = sns.barplot(x = bins_value, y= effective, color = sns.color_palette()[0], width = 0.9, ax = axs[1])
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ymax_value = max(max(emp), max(effective)) 
    ax1.set_ylim(0, ymax_value)
    ax2.set_ylim(0, ymax_value)
    None

def cs_emp_effective_we_std_hist_plot(emp, effective, w, std):
     
    fig, axs = plt.subplots(2, 2 , figsize = (15, 8))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 31, 1)

    ax1 = sns.barplot(x = bins_value, y= emp, color = sns.color_palette()[0], width = 0.9, ax = axs[0, 0])
    ax1.set_title("Emprical Label Distribution", fontsize = 14)
    ax2 = sns.barplot(x = bins_value, y= effective, color = sns.color_palette()[0], width = 0.9, ax = axs[0, 1])
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ymax_value = max(max(emp), max(effective)) 
    ax1.set_ylim(0, ymax_value)
    ax2.set_ylim(0, ymax_value)

    ax3 = sns.barplot(x = bins_value, y= w, color = sns.color_palette()[0], width = 0.9, ax = axs[1, 0])
    ax3.set_title("Weights", fontsize = 14)
    ax4 = sns.barplot(x = bins_value, y= std, color = sns.color_palette()[0], width = 0.9, ax = axs[1, 1])
    ax4.set_title("STD of bin pixels value", fontsize = 14)


    None

def aggregate2(src, scale):
    
    w = int(src.shape[0]/scale)
    h = int(src.shape[1]/scale)
    mtx = np.full((w, h), -1, dtype=np.float32)
    for i in range(w):
        for j in range(h):   
            mtx[i,j]=np.mean(src[i*scale:(i+1)*scale, j*scale:(j+1)*scale])                    
    return mtx    

def triple_emp_effective_weights_hist_plot(emp, effective, weights, method: str):
     
    fig, axs = plt.subplots(1, 3 , figsize = (21, 4))

    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace = 0.01)

    bins_value  = np.arange(1, 70, 1)
    ax1 = sns.barplot(x = bins_value, y = emp, color = sns.color_palette()[0],  ax = axs[0]) #width = 0.9,
    ax1.set_title("Emprical Label Distribution", fontsize = 14)

    ax2 = sns.barplot(x = bins_value, y = effective, color = sns.color_palette()[0], ax = axs[1]) #width = 0.9, 
    ax2.set_title("Effective Label Distribution", fontsize = 14)

    ax3 = sns.barplot(x = bins_value, y = weights, color = sns.color_palette()[0],  ax = axs[2]) #width = 0.9,
    ax3.set_title(r" Weights ({})".format(method), fontsize = 14)

    None

class vineyard_data_visulization():
    def __init__(self, dataset:dict, cultivar: list):
        self.dataset = dataset
        self.cultivar  = cultivar
    
    
    def by_cultivar_label_dist_vis(self): 


        this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == self.cultivar[0]}


        fig, axs = plt.subplots(1, 2 , figsize = (12, 4))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []
        concat_matrix_all_blocks = []


        for k, v in this_cultivar_dict.items():
    
            label_mtx = v['label']
            label_mtx = label_mtx.flatten()
            label_mtx = label_mtx[label_mtx >= 0] 
            concat_matrix_all_blocks.append(label_mtx)


            ax1 = sns.kdeplot(label_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)

        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles, loc = 'upper left')
        ax1.set(ylabel='Density')


        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle(self.cultivar[0] + ': label distribution')

    def by_cultivar_image_dist_vis(self): 


        this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == self.cultivar[0]}

        fig, axs = plt.subplots(1, 2 , figsize = (12, 4))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []
        for k, v in this_cultivar_dict.items():
            img_mtx = v['image']
            img_mtx = np.nan_to_num(img_mtx)
            img_mtx = img_mtx.flatten()

            ax1 = sns.kdeplot(img_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)
        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles, loc = 'upper left')
        ax1.set(ylabel='Density')

        concat_matrix_all_blocks = []
        for k, v in this_cultivar_dict.items():
            concat_matrix_all_blocks.append(v['image'].flatten())

        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks, axis = 0)
        concat_matrix_all_blocks = np.nan_to_num(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle(self.cultivar[0] + ': feature distribution')

    def all_cultivar_label_dist_vis(self): 


        new_dict = {}

        for cul in self.cultivar: 
            this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == cul}
            labels = []
            for k2, v2 in this_cultivar_dict.items(): 
                labels.append(v2['label'].flatten())
            labels = np.concatenate(labels)
            labels = labels[labels >= 0]

            new_dict[cul] = {'label': labels} 


        fig, axs = plt.subplots(1, 2 , figsize = (16, 6))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []
        concat_matrix_all_blocks = []


        for k, v in new_dict.items():
            label_mtx = v['label']
            concat_matrix_all_blocks.append(label_mtx)
            ax1 = sns.kdeplot(label_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)

        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles, loc = 'upper left')
        ax1.set(ylabel='Density')

        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle('All cultivars label distribution')


    def all_cultivar_image_dist_vis(self): 

        new_dict = {}

        for cul in self.cultivar: 
            this_cultivar_dict = {k:v for k, v in self.dataset.items() if self.dataset[k]['cultivar'] == cul}
            images = []
            for k2, v2 in this_cultivar_dict.items(): 
                images.append(v2['image'].flatten())
            images = np.concatenate(images)
            images = np.nan_to_num(images)

            new_dict[cul] = {'image': images} 

        fig, axs = plt.subplots(1, 2 , figsize = (14, 5))

        sns.set_style("whitegrid", {'axes.grid' : False})
        plt.rcParams["figure.autolayout"] = True
        plt.subplots_adjust(hspace = 0.01)


        plot_labels, handles = [], []

        for k, v in new_dict.items():
            img_mtx = v['image']
            ax1 = sns.kdeplot(img_mtx, ax = axs[0], palette='Dark2', shade ='fill', legend=True,)
            plot_labels.append(k)

        for i in range(len(plot_labels)):
            handles.append(mpatches.Patch(facecolor = sns.color_palette()[i], label = plot_labels[i]))

        axs[0].legend(handles = handles)
        ax1.set(ylabel='Density')

        concat_matrix_all_blocks = []
        for k, v in new_dict.items():
            concat_matrix_all_blocks.append(v['image'])

        concat_matrix_all_blocks = np.concatenate(concat_matrix_all_blocks)

        ax2 = sns.kdeplot(concat_matrix_all_blocks, ax = axs[1], palette='Dark2', shade ='fill')

        fig.suptitle('All cultivars feature distribution')

def block_true_pred_mtx(df, block_id, aggregation = None, spatial_resolution  = None, scale = None):
    
    name_split = os.path.split(str(block_id))[-1]
    root_name  = name_split.replace(name_split[-4:], '')
    year       = name_split[-4:]
    
    if len(root_name) == 1:
        this_block_name = 'LIV_00' + root_name + '_' + year
    elif len(root_name) == 2:
        this_block_name = 'LIV_0' + root_name + '_' + year
    elif len(root_name) == 3:
        this_block_name = 'LIV_' + root_name + '_' + year
    
    #print(this_block_name)
    blocks_df = df.groupby(by = 'block')
    this_block_df = blocks_df.get_group(block_id)
    
    
    
    res           = {key: blocks_size[key] for key in blocks_size.keys() & {this_block_name}}
    list_d        = res.get(this_block_name)
    block_x_size  = int(list_d[0]/spatial_resolution)
    block_y_size  = int(list_d[1]/spatial_resolution)
    
    print(this_block_df.shape)
    pred_out = np.full((block_x_size, block_y_size), -1) 
    true_out = np.full((block_x_size, block_y_size), -1)  

    for x in range(block_x_size):
        for y in range(block_y_size):
            
            new            = this_block_df.loc[(this_block_df['x'] == x)&(this_block_df['y'] == y)] 
            if len(new) > 0:
                pred_out[x, y] = new['ypred_w15'].min()*2.2417
                true_out[x, y] = new['ytrue'].mean()*2.2417


    if aggregation is True: 
        
        print(f"{pred_out.shape}|{true_out.shape}")
        pred_agg = aggregate2(pred_out, scale)
        true_agg = aggregate2(true_out, scale)
        print(f"Agg: {pred_agg.shape}|{true_agg.shape}")
        
        df_agg = pd.DataFrame() 
        
        flat_pred_agg = pred_agg.flatten()
        flat_pred_agg = flat_pred_agg[flat_pred_agg != -1]
        
        flat_true_agg = true_agg.flatten()
        flat_true_agg = flat_true_agg[flat_true_agg != -1] 
              
        df_agg['ytrue'] = flat_true_agg
        df_agg['ypred'] = flat_pred_agg 
        
        return df_agg, true_agg, pred_agg
    else:
        return this_block_df, true_out, pred_out            

def image_mae_mape_map(ytrue, ypred): 

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
        elif abs(ytrue_flat[i] - ypred_flat[i]) > 11:
            out_mae[i]  = -10
            out_mape[i] = -10
        else: 
            out_mae[i]  = abs(ytrue_flat[i] - ypred_flat[i])
            out_mape[i] = ((abs(ytrue_flat[i] - ypred_flat[i]))/ytrue_flat[i])*100

    out1 = out_mae.reshape(ytrue.shape[0], ytrue.shape[1])
    out2 = out_mape.reshape(ytrue.shape[0], ytrue.shape[1])

    return out1, out2

def yield_true_pred_plot(ytrue, ypred, min_v = None, max_v= None):

    #from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    #mae_map = image_subtract(ytrue, ypred) 
    #mape_map = image_subtract(ytrue, ypred) 
    mae_map, mape_map = image_mae_mape_map(ytrue, ypred)
    img3 = axs[2].imshow(mae_map, cmap = 'viridis') #, cmap = 'magma'
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[2].set_title('MAE Map (t/h)', fontsize = 14)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar2 =fig.colorbar(img3, cax=cax)
    img3.set_clim(-1, np.max(mae_map))
    axs[2].get_yaxis().set_visible(False)
    #axs[2].get_xaxis().set_visible(False)
    #axs[2].set_xlabel(xlabel_text)

    img4 = axs[3].imshow(mape_map, cmap = 'viridis') #
    #xlabel_text = (r"($PSNR = {:.2f}$" + ", " + r"$SSIM = {:.2f}$)").format(PSNR, ssim_value)
    axs[3].set_title('MAPE Map (%)', fontsize = 14)
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar3 =fig.colorbar(img4, cax=cax)
    img4.set_clim(-5, 20)
    axs[3].get_yaxis().set_visible(False)

    #return mape_map
    #plt.savefig('./imgs/B186_1m.png', dpi = 300)

class data_generator():
    def __init__(self, eval_scenario: str, 
                    spatial_resolution: int, 
                    patch_size: int, 
                    patch_offset: int,  
                    cultivar_list: list, 
                    year_list: list):

        self.eval_scenario      = eval_scenario
        self.spatial_resolution = spatial_resolution
        self.patch_size         = patch_size
        self.patch_offset       = patch_offset
        self.cultivar_list      = cultivar_list
        self.year_list          = year_list


        if self.cultivar_list is None: 
            self.cultivar_list = ['MALVASIA_BIANCA', 'MUSCAT_OF_ALEXANDRIA', 
                                    'CABERNET_SAUVIGNON','SYMPHONY', 
                                    'MERLOT', 'CHARDONNAY', 
                                    'SYRAH', 'RIESLING']

        if self.spatial_resolution == 1: 
            self.npy_dir = '/data2/hkaman/Livingston/data/1m/'

        else: 
            self.npy_dir = '/data2/hkaman/Livingston/data/10m/'

        self.images_dir  = os.path.join(self.npy_dir, 'imgs')
        self.image_names = os.listdir(self.images_dir)
        self.image_names.sort() 

        self.label_dir   = os.path.join(self.npy_dir, 'labels')
        self.label_names = os.listdir(self.label_dir)
        self.label_names.sort() 


    def return_split_dataframe(self):

        full_dataframe = self.return_dataframe_patch_info()


        if self.eval_scenario == 'pixel_hold_out': 
            train, valid, test = self.pixel_hold_out(full_dataframe)
        elif self.eval_scenario == 'year_hold_out':
            train, valid, test = self.year_hold_out(full_dataframe)
        elif self.eval_scenario == 'block_hold_out': 
            train, valid, test = self.block_hold_out(full_dataframe)
        elif self.eval_scenario == 'block_year_hold_out': 
            train, valid, test = self.block_year_hold_out(full_dataframe)

        '''print(f"Training Patches: {len(train)}, Validation: {len(valid)} and Test: {len(test)}")
        print("============================= Train =========================================")
        _ = print_df_summary(train)
        print("============================= Validation ====================================")
        _ = print_df_summary(valid)
        print("============================= Test ==========================================")
        _ = print_df_summary(test)
        print("=============================================================================")'''

        return train, valid, test


    def return_dataframe_patch_info(self): 

        df = pd.DataFrame()

        Block, Cultivar, CID, Trellis, TID, RW, SP = [], [], [], [], [], [], []
        P_means, YEAR, X_COOR, Y_COOR, IMG_P, Label_P  = [], [], [], [], [], []
        
        
        generated_cases = 0
        removed_cases = 0 
        
        
        for idx, name in enumerate(self.label_names):
            # Extract Image path
            image_path = os.path.join(self.images_dir, self.image_names[idx])

            name_split  = os.path.split(name)[-1]
            block_name  = name_split.replace(name_split[12:], '')
            root_name   = name_split.replace(name_split[7:], '')
            year        = name_split.replace(name_split[0:8], '').replace(name_split[12:], '')
            
            res           = {key: configs.blocks_information[key] for key in configs.blocks_information.keys() & {root_name}}
            list_d        = res.get(root_name)
            block_variety = list_d[0]
            block_id      = list_d[1]
            block_rw      = list_d[2]
            block_sp      = list_d[3]
            block_trellis = list_d[5]
            block_tid     = list_d[6]

            label_npy = os.path.join(self.label_dir, name)
            label = np.load(label_npy, allow_pickle=True)
            label = label[0,:,:,0]
            width, height = label.shape[1], label.shape[0]
            
            
            for i in range(0, height - self.patch_size, self.patch_offset):
                for j in range(0, width - self.patch_size, self.patch_offset):
                    crop_label = label[i:i+ self.patch_size, j:j+ self.patch_size]
                    
                    if np.any((crop_label < 0)):
                        removed_cases += 1
                        
                    elif np.all((crop_label >= 0)): 

                        generated_cases += 1
                        
                        patch_mean       = np.mean(crop_label)
                        P_means.append(patch_mean)
                    
                        Block.append(block_name)
                        CID.append(block_id)
                        Cultivar.append(block_variety)
                        Trellis.append(block_trellis)
                        TID.append(block_tid)
                        RW.append(int(block_rw))
                        SP.append(int(block_sp))
                        YEAR.append(year)
                        X_COOR.append(i)
                        Y_COOR.append(j)
                        IMG_P.append(image_path)
                        Label_P.append(label_npy)                                        

                        
        df['block']       = Block
        df['X']           = X_COOR
        df['Y']           = Y_COOR
        df['year']        = YEAR
        df['cultivar_id'] = CID
        df['cultivar']    = Cultivar
        df['trellis']     = Trellis
        df['trellis_id']  = TID
        df['row']         = RW
        df['space']       = SP
        df['patch_mean']     = P_means
        df['IMG_PATH']    = IMG_P
        df['LABEL_PATH']  = Label_P
        
        if self.cultivar_list is None:
            Customized_df = df
            
        else: 
            Customized_df = df[df['cultivar'].isin(self.cultivar_list)]
            
        return Customized_df


    def year_hold_out(self, df): 

        NewGroupedDf = df.groupby(by=["year"])

        Group1 = NewGroupedDf.get_group(self.year_list[0])
        Group2 = NewGroupedDf.get_group(self.year_list[1])
        Group3 = NewGroupedDf.get_group(self.year_list[2])
        Group4 = NewGroupedDf.get_group(self.year_list[3])

        frames = [Group1, Group2]
        train = pd.concat(frames)
        valid = Group3
        test  = Group4

        return train, valid, test
    
    def block_hold_out(self, df):
        
        datafram_grouby_year = df.groupby(by = 'year')
        dataframe_year2017   = datafram_grouby_year.get_group('2017')
        
        new_dataframe_basedon_block_mean = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        dataframe_year2017_groupby_block = dataframe_year2017.groupby(by = 'block')

        for block, blockdf in dataframe_year2017_groupby_block:
            name_split = os.path.split(block)[-1]
            root_name  = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        new_dataframe_basedon_block_mean['block'] = block_root_name
        new_dataframe_basedon_block_mean['cultivar'] = cultivar
        new_dataframe_basedon_block_mean['block_mean'] = b_mean
            
        # split sorted blocks and then split within each cultivar 
        BlockMeanBased_GroupBy_Cultivar = new_dataframe_basedon_block_mean.groupby(by=["cultivar"]) 
        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in BlockMeanBased_GroupBy_Cultivar: 
            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if n_blocks <= 1: 
                name_2016  = frame['block'].iloc[0] + '_2016'
                name_2017  = frame['block'].iloc[0] + '_2017'
                name_2018  = frame['block'].iloc[0] + '_2018'
                name_2019  = frame['block'].iloc[0] + '_2019'
                training_blocks_names.extend((name_2016, name_2017, name_2018, name_2019))
                
            elif n_blocks == 2:
                name_2016_0  = frame['block'].iloc[0] + '_2016'
                name_2017_0  = frame['block'].iloc[0] + '_2017'
                name_2018_0  = frame['block'].iloc[0] + '_2018'
                name_2019_0  = frame['block'].iloc[0] + '_2019'
                
                training_blocks_names.extend((name_2016_0, name_2017_0, name_2018_0, name_2019_0))
                
                name_2016_1  = frame['block'].iloc[1] + '_2016'
                name_2017_1  = frame['block'].iloc[1] + '_2017'
                name_2018_1  = frame['block'].iloc[1] + '_2018'
                name_2019_1  = frame['block'].iloc[1] + '_2019'
                
                validation_blocks_names.extend((name_2016_1, name_2017_1, name_2018_1, name_2019_1))
                
            elif n_blocks == 3:
                name_2016_0  = frame['block'].iloc[0] + '_2016'
                name_2017_0  = frame['block'].iloc[0] + '_2017'
                name_2018_0  = frame['block'].iloc[0] + '_2018'
                name_2019_0  = frame['block'].iloc[0] + '_2019'
                
                training_blocks_names.extend((name_2016_0, name_2017_0, name_2018_0, name_2019_0))
                
                name_2016_1  = frame['block'].iloc[2] + '_2016'
                name_2017_1  = frame['block'].iloc[2] + '_2017'
                name_2018_1  = frame['block'].iloc[2] + '_2018'
                name_2019_1  = frame['block'].iloc[2] + '_2019'
                
                testing_blocks_names.extend((name_2016_1, name_2017_1, name_2018_1, name_2019_1))  
                
                name_2016_2  = frame['block'].iloc[1] + '_2016'
                name_2017_2  = frame['block'].iloc[1] + '_2017'
                name_2018_2  = frame['block'].iloc[1] + '_2018'
                name_2019_2  = frame['block'].iloc[1] + '_2019'
                
                validation_blocks_names.extend((name_2016_2, name_2017_2, name_2018_2, name_2019_2)) 
                
            elif n_blocks > 3:
                blocks_2017      = frame['block']
                blocks_mean_2017 = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield_2017 = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_2017, blocks_mean_2017)]

                block_mean_yield_2017 = sorted(block_mean_yield_2017, key = lambda x: x[1], reverse = True)
 

                te  = 1
                val = 2
                for i in range(len(block_mean_yield_2017)):
                    name_2016  = block_mean_yield_2017[i][0] + '_2016'
                    name_2017  = block_mean_yield_2017[i][0] + '_2017'
                    name_2018  = block_mean_yield_2017[i][0] + '_2018'
                    name_2019  = block_mean_yield_2017[i][0] + '_2019'

                    if i == te: 
                        testing_blocks_names.append(name_2016)
                        testing_blocks_names.append(name_2017)
                        testing_blocks_names.append(name_2018)
                        testing_blocks_names.append(name_2019)
                        te = te + 3
                    elif i == val: 
                        validation_blocks_names.append(name_2016)
                        validation_blocks_names.append(name_2017)
                        validation_blocks_names.append(name_2018)
                        validation_blocks_names.append(name_2019)

                        val = val + 3
                    else:
                        training_blocks_names.append(name_2016)
                        training_blocks_names.append(name_2017)
                        training_blocks_names.append(name_2018)
                        training_blocks_names.append(name_2019)

        train = df[df['block'].isin(training_blocks_names)]
        valid = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)] 


        return train, valid, test

    def block_year_hold_out(self, df):
        
        datafram_grouby_year = df.groupby(by = 'year')
        dataframe_year2017   = datafram_grouby_year.get_group('2017')
        
        new_dataframe_basedon_block_mean = pd.DataFrame()
        block_root_name, cultivar, b_mean = [], [], []
        
        dataframe_year2017_groupby_block = dataframe_year2017.groupby(by = 'block')

        for block, blockdf in dataframe_year2017_groupby_block:
            name_split = os.path.split(block)[-1]
            root_name  = name_split.replace(name_split[7:], '')
            block_root_name.append(root_name)
            
            cultivar.append(blockdf['cultivar'].iloc[0])
            b_mean.append(blockdf['patch_mean'].mean())
            
        new_dataframe_basedon_block_mean['block']      = block_root_name
        new_dataframe_basedon_block_mean['cultivar']   = cultivar
        new_dataframe_basedon_block_mean['block_mean'] = b_mean
        
        # split sorted blocks and then split within each cultivar 
        BlockMeanBased_GroupBy_Cultivar = new_dataframe_basedon_block_mean.groupby(by=["cultivar"]) 

        training_blocks_names = []
        validation_blocks_names = []
        testing_blocks_names = []
        
        for cul, frame in BlockMeanBased_GroupBy_Cultivar: 

            n_blocks = len(frame.loc[frame['cultivar'] == cul])
            
            if frame.shape[0] == 3:
                

                name_0  = frame['block'].iloc[0] + '_' + self.year_list[0]
                name_1  = frame['block'].iloc[0] + '_' + self.year_list[1]
                training_blocks_names.append(name_0)
                training_blocks_names.append(name_1)
                
                name_2  = frame['block'].iloc[1] + '_' + self.year_list[2]
                validation_blocks_names.append(name_2) 

                name_3  = frame['block'].iloc[2] + '_' + self.year_list[3]
                testing_blocks_names.append(name_3) 


                
            elif frame.shape[0] > 3:

                blocks_2017      = frame['block']
                blocks_mean_2017 = frame['block_mean']

                # List of tuples with blocks and mean yield
                block_mean_yield_2017 = [(blocks, mean) for blocks, 
                                    mean in zip(blocks_2017, blocks_mean_2017)]

                block_mean_yield_2017 = sorted(block_mean_yield_2017, key = lambda x: x[1], reverse = True)
                #print(block_mean_yield_2017)
                #print("============================")

                te  = 1
                val = 2
                for i in range(len(block_mean_yield_2017)):

                    if i == te: 
                        name_3  = block_mean_yield_2017[i][0] + '_' + self.year_list[3]
                        testing_blocks_names.append(name_3)
                        te = te + 3
                        #print(f"{cul}: {name_3}")
                    elif i == val: 
                        name_2  = block_mean_yield_2017[i][0] + '_' + self.year_list[2]
                        validation_blocks_names.append(name_2)
                        val = val + 3
                        #print(f"{cul}: {name_2}")
                    else:
                        name_0  = block_mean_yield_2017[i][0] + '_' + self.year_list[0]
                        name_1  = block_mean_yield_2017[i][0] + '_' + self.year_list[1]
                        #print(f"{cul}: {name_0, name_1}")
                        training_blocks_names.append(name_0)
                        training_blocks_names.append(name_1)
                    #print(f"with MORE than 3: {name_0, name_1, name_2, name_3}")

        #print(validation_blocks_names)
        train = df[df['block'].isin(training_blocks_names)]
        valid = df[df['block'].isin(validation_blocks_names)]
        test  = df[df['block'].isin(testing_blocks_names)] 


        return train, valid, test


    def return_pixelwise_weight_dw(self, dw_alpha):

        masks = None
        for idx, row in self.NewDf.iterrows():
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

        #weights = TargetRelevance(reshaped_masks, alpha = dw_alpha).get_relevance()
        weights = TargetRelevance(reshaped_masks, alpha = dw_alpha).__call__(reshaped_masks)

        weights = np.reshape(weights, (masks.shape[0], masks.shape[1], masks.shape[2]))
        return weights  
#-------------------------------------------------------------------------------------------#
#                                Data Complexity Measures                                   #
#-------------------------------------------------------------------------------------------# 
class data_complexity_measure():
    def __init__(self, dataframe, proj: str, method: str, basedon: str):

        self.dataset = dataframe
        self.proj    = proj
        self.method  = method 
        self.basedon = basedon

    def return_output(self):
        if self.method == 'F1':
            output = self.MaxFisherDR()

        return output

    def MaxFisherDR(self):
        if self.proj == 'vienyard':
            unique_classes = sorted(self.dataset['cultivar'].unique())

        elif self.proj == 'lettuce':
            unique_classes = sorted(self.dataset['LtID'].unique())

        cls = []
        f1s=[]

        for i in unique_classes:
            for j in unique_classes: 
                if (i != j) and (j > i): 
                    #get the samples of the 2 classes being considered this iteration
                    sample_c1 = df.loc[df['LtID'] == i]
                    sample_c2 = df.loc[df['LtID'] == j]

                    avg_c1 = np.mean(sample_c1['biomass'], 0)
                    avg_c2 = np.mean(sample_c2['biomass'], 0)

                    std_c1 = np.std(sample_c1['biomass'], 0)
                    std_c2 = np.std(sample_c2['biomass'], 0)

                    f1 = ((avg_c1-avg_c2)**2)/(std_c1**2+std_c2**2)

                    f1 = 1/(1+f1)
                    f1s.append(f1)
                    this_combination = r'LtID: {} vs {}'.format(i, j)
                    cls.append(this_combination)

        list_zip = list(zip(cls, f1s))
        f1_val = np.mean(f1s,axis=0)
    
        return f1_val


    def F2(self):
        '''
        Calculates the F2 measure defined in [1]. 
        Uses One vs One method to handle multiclass datasets.
        -----
        Returns:
        f2s (array): The value of f2 measure for each one vs one combination of classes.
        -----
        References:
        [1] Lorena AC, Garcia LP, Lehmann J, Souto MC, Ho TK (2019) How com-
        plex is your classification problem? a survey on measuring classification
        complexity. ACM Computing Surveys (CSUR) 52(5):1-34
        '''
        f2s=[]
        #one vs one method
        for i in range(len(self.class_inxs)):
            for j in range(i+1,len(self.class_inxs)):
                sample_c1 = self.X[self.class_inxs[i]]
                sample_c2 = self.X[self.class_inxs[j]]

                maxmax = np.max([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)
                maxmin = np.max([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmin = np.min([np.min(sample_c1,axis=0),np.min(sample_c2,axis=0)],axis=0)
                minmax = np.min([np.max(sample_c1,axis=0),np.max(sample_c2,axis=0)],axis=0)

                numer=np.maximum(0.0, minmax - maxmin) 
            
                denom=(maxmax - minmin)
        
                f2 = np.prod(numer/denom)
                f2s.append(f2)

        return f2s

