import os
import numpy as np
import pandas as pd
import torch
from PyPDF2 import PdfReader
import re
import numpy as np
import cv2
from models.configs import set_seed
set_seed(1987)

EXTREME_LOWER_THRESHOLD = 9  #22.24
EXTREME_UPPER_THRESHOLD = 22 #54.36
HECTARE_TO_ACRE_SCALE = 2.471 # 2.2417


def cost_sensitive_weight_sampler(df):
    
    Groups = df.groupby(by=["cultivar"])
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    dict_ = {}
    for state, frame in Groups:        
        count_list = frame['patch_mean'].value_counts(bins=bins, sort=False)
        count_sum = np.sum(count_list)
        dict_[state[0]] = count_list, count_sum

    weight = []#np.zeros((len(df))) 
    
    for idx, row in df.iterrows():  
        patch_cultivar = row['cultivar']
        patch_mean = row['patch_mean']     

        get_patch_count = dict_[patch_cultivar][0][patch_mean]
        get_cultivar_sum = dict_[patch_cultivar][1]
        row_weight = get_patch_count / get_cultivar_sum
        # row_weight = 1 / (get_patch_count / get_patch_count) if get_patch_count != 0 else 0
        weight.append(row_weight)
        
    weight = np.array(weight)
    df['weight'] = weight
    list_sum = df.groupby(by=["cultivar"])['weight'].transform('sum')
    NormWeights = df['weight']/list_sum
    # df['NWeight'] = NormWeights
    
    return NormWeights

def cost_sensitive_weight_sampler2(df):
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    # df['label_mean'] = df['label_path'].apply(lambda path: np.mean(np.array(Image.open(path))))
    count_list = df['patch_mean'].value_counts(bins=bins, sort=False)
    count_sum = np.sum(count_list)
    
    dict_ = {
        "all": (count_list, count_sum)
    }
    
    weight = [] 
    for idx, row in df.iterrows():
        label_mean = row['patch_mean']
        get_label_count = dict_["all"][0][label_mean]
        get_total_sum = dict_["all"][1]
        
        # Use logarithmic scaling for weights
        row_weight = np.log(1 + get_total_sum / get_label_count) if get_label_count != 0 else 0
        weight.append(row_weight)
    
    weight = np.array(weight)
    df['weight'] = weight
    total_weight_sum = df['weight'].sum()
    NormWeights = df['weight'] / total_weight_sum
    
    return NormWeights

def cost_sensitive_weight_sampler_3(df):
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    # Group by cultivar for cultivar-specific weights
    Groups = df.groupby(by=["cultivar"])
    dict_ = {}
    
    for cultivar, frame in Groups:
        count_list = frame['patch_mean'].value_counts(bins=bins, sort=False)
        count_sum = np.sum(count_list)
        dict_[cultivar[0]] = count_list, count_sum

    weight = [] 
    for idx, row in df.iterrows():
        patch_cultivar = row['cultivar']
        patch_mean = row['patch_mean']
        
        # Get count for the patch mean and total for the cultivar
        get_label_count = dict_[patch_cultivar][0][patch_mean]
        get_total_sum = dict_[patch_cultivar][1]
        
        # Apply logarithmic scaling to avoid extreme values
        row_weight = np.log(1 + get_total_sum / get_label_count) if get_label_count != 0 else 0
        weight.append(row_weight)

    # Normalize weights across all samples
    weight = np.array(weight)
    df['weight'] = weight
    total_weight_sum = df['weight'].sum()
    NormWeights = df['weight'] / total_weight_sum
    
    return NormWeights

def cost_sensitive_weight_sampler_optimized(df):
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    
    # Group by "cultivar" if needed, otherwise use the entire dataframe
    Groups = df.groupby(by=["cultivar"])
    
    dict_ = {}
    for cultivar, frame in Groups:
        count_list = frame['patch_mean'].value_counts(bins=bins, sort=False)
        count_sum = np.sum(count_list)
        dict_[cultivar[0]] = count_list, count_sum

    weight = [] 
    for idx, row in df.iterrows():
        patch_cultivar = row['cultivar']
        patch_mean = row['patch_mean']
        
        # Fetch the patch count and sum for the corresponding cultivar
        get_patch_count = dict_[patch_cultivar][0][patch_mean]
        get_cultivar_sum = dict_[patch_cultivar][1]
        
        # Safely calculate row weight with logarithmic scaling
        if get_patch_count > 0:
            row_weight = np.log(1 + get_cultivar_sum / get_patch_count)
        else:
            row_weight = 0  # Handle the case where the count is zero
        
        weight.append(row_weight)
    
    weight = np.array(weight)
    df['weight'] = weight
    
    # Normalize the weights for each cultivar to sum to 1
    df['NormWeights'] = df.groupby('cultivar')['weight'].transform(lambda x: x / x.sum())

    return df['NormWeights']


def get_dataloaders(
        batch_size:int, 
        img_size: int,
        in_channels:int, 
        resmapling_status: False,
        data: str,
        exp_name: str): 

    root_data_dir = '/data2/hkaman/Data/'
    root_exp_dir = '/data2/hkaman/Projects/'

    exp_output_dir = root_exp_dir + 'ViT/EXPs/Sep/' + 'EXP_' + exp_name

    isExist  = os.path.isdir(exp_output_dir)

    if not isExist:
        os.makedirs(exp_output_dir)
        os.makedirs(os.path.join(exp_output_dir, 'checkpoints'))
        os.makedirs(os.path.join(exp_output_dir, 'coords'))
        os.makedirs(os.path.join(exp_output_dir, 'loss'))
        os.makedirs(os.path.join(exp_output_dir, 'attn_scores'))

    if data == 's2':
        train_csv = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/train.csv', index_col=0)
        train_csv.to_csv(os.path.join(exp_output_dir + '/coords','train.csv'))
        valid_csv = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/val.csv', index_col= 0)
        valid_csv.to_csv(os.path.join(exp_output_dir + '/coords','val.csv'))
        test_csv  = pd.read_csv('/data2/hkaman/Data/Coords/S2/BHO/test.csv', index_col= 0)
        test_csv.to_csv(os.path.join(exp_output_dir + '/coords','test.csv'))

    elif data == 'p':
        train_csv = pd.read_csv('/data2/hkaman/Data/Coords/Planet/BHO/train.csv', index_col=0)
        train_csv.to_csv(os.path.join(exp_output_dir + '/coords','train.csv'))
        valid_csv = pd.read_csv('/data2/hkaman/Data/Coords/Planet/BHO/val.csv', index_col= 0)
        valid_csv.to_csv(os.path.join(exp_output_dir + '/coords','val.csv'))
        test_csv  = pd.read_csv('/data2/hkaman/Data/Coords/Planet/BHO/test.csv', index_col= 0)
        test_csv.to_csv(os.path.join(exp_output_dir + '/coords','test.csv'))

    print(f"{train_csv.shape} | {valid_csv.shape} | {test_csv.shape}")
    #==============================================================================================================#
    #============================================     Reading Data                =================================#
    #==============================================================================================================#
    #csv_coord_dir = '/data2/hkaman/Livingston/EXPs/10m/EXP_S3_UNetLSTM_10m_time/'

    if data == 's2':
        data_dir = root_data_dir + 'Livingston/data/10m/'
    elif data =='p':
        data_dir = root_data_dir + 'planet/data/'


    dataset_training = DataCreator(
        data_dir, 
        exp_output_dir, 
        category = 'train', 
        patch_size = img_size, 
        in_channels = in_channels,
    )

    dataset_validate = DataCreator(
        data_dir, 
        exp_output_dir, 
        category = 'val',  
        patch_size = img_size, 
        in_channels = in_channels,
    )
    
    dataset_test = DataCreator(
        data_dir, 
        exp_output_dir, 
        category = 'test',  
        patch_size = img_size, 
        in_channels = in_channels,
    )     

    #==============================================================================================================#
    #=============================================      Data Loader               =================================#
    #==============================================================================================================#                      
    # define training and validation data loaders
    if resmapling_status is True: 

        if resmapling_status: 
            print(f"resampling is {resmapling_status}, The dataloader is processing cost-sensitive resampling!")

        train_weights = train_csv['NormWeight'].to_numpy() #
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        train_weights, 
        len(train_weights), 
        replacement=True)    

        val_weights   = valid_csv['NormWeight'].to_numpy()
        val_weights   = torch.DoubleTensor(val_weights)
        val_sampler   = torch.utils.data.sampler.WeightedRandomSampler(
        val_weights, 
        len(val_weights), 
        replacement=True)    
    
        data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                        shuffle=False,  sampler=train_sampler, num_workers=8)  
        data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                        shuffle=False, sampler= val_sampler, num_workers=8) 
        data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                        shuffle=False, num_workers=8)  
    else: 
        data_loader_training = torch.utils.data.DataLoader(dataset_training, batch_size= batch_size, 
                                                        shuffle=True,  num_workers=8) 
        data_loader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size= batch_size, 
                                                        shuffle=False, num_workers=8)  
        data_loader_test     = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                        shuffle=False, num_workers=8) 

    return data_loader_training, data_loader_validate, data_loader_test

class DataCreator(object):
    def __init__(self, npy_dir, csv_dir, 
                                category: str, 
                                patch_size: int, 
                                in_channels: int, 
                                ):

        self.npy_dir      = npy_dir
        self.csv_dir      = csv_dir
        self.wsize        = patch_size
        self.in_channels  = in_channels

        if category    == 'train': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/train.csv', index_col=0) 
            self.NewDf.reset_index(inplace = True, drop = True)

        elif category  == 'val': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/val.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)

        elif category  == 'test': 
            self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/test.csv', index_col=0)
            self.NewDf.reset_index(inplace = True, drop = True)

        self.weights = self.return_pixelwise_weight_dw(3.9)
        assert not np.isnan(self.weights).any()
        self.weights = np.where(self.weights >= 1, self.weights, 1)

        self.stats_dict = np.load('/data2/hkaman/Data/Coords/met_stats.npz', allow_pickle=True)
        self.stats_dict = self.stats_dict['arr_0'].item() 

    def __getitem__(self, idx):

        xcoord = self.NewDf.loc[idx]['X'] 
        ycoord = self.NewDf.loc[idx]['Y'] 
        block_id = self.NewDf.loc[idx]['block']
        cultivar = self.NewDf.loc[idx]['cultivar']
        cultivar_id = self.NewDf.loc[idx]['cultivar_id']
        rw_id = self.NewDf.loc[idx]['row']
        sp_id = self.NewDf.loc[idx]['space']
        t_id = self.NewDf.loc[idx]['trellis_id']



        if self.in_channels == 4:
            # only Sentinel 1 and time encoding
            WithinBlockMean = self.NewDf.loc[idx]['win_block_mean']
            block_means = self.add_input_within_bc_mean(WithinBlockMean)
            block_timeseries_encode = self._time_series_encoding(block_id)
            S1_path = self.NewDf.loc[idx]['S1_PATH']
            S1 = self._crop_gen(S1_path, xcoord, ycoord) 
            S1 = np.swapaxes(S1, -1, 0)
            image = np.concatenate([S1, block_timeseries_encode, block_means], axis = 0)

        elif self.in_channels == 6:
            S2_path = self.NewDf.loc[idx]['IMG_PATH']
            S2 = self._crop_gen(S2_path, xcoord, ycoord) 
            S2 = np.swapaxes(S2, -1, 0)   
            S2 = self.histogram_equalization_4d(S2)
            S2 = S2 / 255.

            block_timeseries_encode = self._time_series_encoding(block_id)
            WithinBlockMean = self.NewDf.loc[idx]['win_block_mean']
            block_means = self.add_input_within_bc_mean(WithinBlockMean)

            image = np.concatenate([S2, block_timeseries_encode, block_means], axis = 0)

        elif self.in_channels == 8: 
            # Sentinel-1 and -2 and time encoding
            S2_path = self.NewDf.loc[idx]['IMG_PATH']
            S2 = self._crop_gen(S2_path, xcoord, ycoord) 
            S2 = np.swapaxes(S2, -1, 0)   
            S2 = self.histogram_equalization_4d(S2)
            S2 = S2 / 255.

            block_timeseries_encode = self._time_series_encoding(block_id)
            WithinBlockMean = self.NewDf.loc[idx]['win_block_mean']
            block_means = self.add_input_within_bc_mean(WithinBlockMean)

            S1_path = self.NewDf.loc[idx]['S1_PATH']
            S1 = self._crop_gen(S1_path, xcoord, ycoord) 
            S1 = np.swapaxes(S1, -1, 0)

            image = np.concatenate([S2, S1, block_timeseries_encode, block_means], axis = 0)

        image = torch.as_tensor(image, dtype=torch.float32)

        # MASK 
        label_path = self.NewDf.loc[idx]['LABEL_PATH']
        mask = self._crop_gen(label_path, xcoord, ycoord) 
        mask = np.swapaxes(mask, -1, 0)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        # YIELDZONE: 
        yz = self._YieldZoneMap(mask, num_classes= 9)
        yz = torch.as_tensor(yz, dtype=torch.float32)

        # Text
        text_path = self.NewDf.loc[idx]['TEXT_PATH']
        EmbText = self._load_text_file(text_path)


        # Meteorological data
        met_path = self.NewDf.loc[idx]['MET_PATH']
        met = np.load(met_path)
        met = self._met_normalizer(met[..., 0], method= 'z-score')

        # Weights
        weight_mtx = self.weights[idx, :, :]
        weight_mtx = np.expand_dims(weight_mtx, axis = 0)
        weight_mtx = torch.as_tensor(weight_mtx, dtype=torch.float32)
        
        sample = {"image": image, 
                  "mask": mask, 
                  "met": met, 
                  "block": block_id, 
                  "cultivar": cultivar, 
                  "X": xcoord, "Y": ycoord, 
                  "EmbList": [cultivar_id, t_id, rw_id, sp_id], 
                  "EmbText": EmbText, 
                  "YZ": yz, 
                  "weight": weight_mtx,} 
            
        return sample

    def __len__(self):
        return len(self.NewDf)
    
    def _load_text_file(self, file_path):
        # Extract the file extension to determine how to process it
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
            # Unsupported file type
            raise ValueError("Unsupported file format: " + file_extension)
    
    def _met_normalizer(self, arr, method='z-score'):
        """
        Normalize the meteorological dataset using either min-max normalization or z-score normalization.

        Parameters
        ----------
        arr : np.ndarray
            The input array to normalize, with the shape (channels, ...).
        stats_dict : dict
            The dictionary containing the statistics (mean, std, min, max) for each variable.
        method : str
            The normalization method, either 'min-max' or 'z-score'.
            
        Returns
        -------
        np.ndarray
            The normalized array.
        """
        for channel in range(1, 4):
            channel_key = {1: 'tmin', 2: 'tmax', 3: 'vp'}[channel]
            
            if method == 'z-score':
                mean_values = np.array(self.stats_dict[channel_key]['mean'])
                std_values = np.array(self.stats_dict[channel_key]['std'])

                for w in range(15):
                    arr[channel, ...] = (arr[channel, ...] - mean_values[w]) / std_values[w]

            elif method == 'min-max':
                min_values = np.array(self.stats_dict[channel_key]['min'])
                max_values = np.array(self.stats_dict[channel_key]['max'])

                for w in range(15):
                    arr[channel, ...] = (arr[channel, ...] - min_values[w]) / (max_values[w] - min_values[w])

        return arr

    def return_yield_zone(self, mask):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented = np.zeros_like(mask)
        # Class 1: Pixel values <= 9
        segmented[mask < EXTREME_LOWER_THRESHOLD] = 1
        # Class 2: Pixel values > 9 and < 22
        segmented[(mask >= EXTREME_LOWER_THRESHOLD) & (mask < EXTREME_UPPER_THRESHOLD)] = 2
        # Class 3: Pixel values >= 22
        segmented[mask >= EXTREME_UPPER_THRESHOLD] = 3

        return segmented
    
    def _YieldZoneMap(self, mask, num_classes: int):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented = np.zeros_like(mask)
        # min_point = 8
        # max_point = 22
        
        if num_classes == 3: 
            # Initialize an empty array with the same shape as the image for the segmented output
            segmented = np.zeros_like(mask)
            # Class 1: Pixel values <= 9
            segmented[mask < EXTREME_LOWER_THRESHOLD] = 1
            # Class 2: Pixel values > 9 and < 22
            segmented[(mask >= EXTREME_LOWER_THRESHOLD) & (mask < EXTREME_UPPER_THRESHOLD)] = 2
            # Class 3: Pixel values >= 22
            segmented[mask >= EXTREME_UPPER_THRESHOLD] = 3

        elif num_classes == 9:
            # Values < 8: Class 1
            segmented[mask < 8] = 1
            
            # Values between 8 and 22: Classes 2 to 8 (7 intervals of 2)
            for i, val in enumerate(range(EXTREME_LOWER_THRESHOLD, EXTREME_UPPER_THRESHOLD, 2), start=2):
                lower_bound = val
                upper_bound = val + 2
                segmented[(mask >= lower_bound) & (mask < upper_bound)] = i
            
            # Values > 22 and < 30: Last class (9)
            segmented[(mask > EXTREME_UPPER_THRESHOLD) & (mask < 30)] = 9
    
            # Value == 30: Also last class (9)
            segmented[mask == 30] = 9

        elif num_classes == 15:
            for i in range(15):
                lower_bound = i * 2
                upper_bound = (i + 1) * 2
                segmented[(mask >= lower_bound) & (mask < upper_bound)] = i + 1
            
            # Special case for the upper boundary of the last class to include the value 30
            segmented[mask == 30] = 15

        return segmented
    
    def return_yield_zone_9_classes(self, mask):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented = np.zeros_like(mask)
        
        # Values < 8: Class 1
        segmented[mask < 8] = 1
        
        # Values between 8 and 22: Classes 2 to 8 (7 intervals of 2)
        for i, val in enumerate(range(8, 22, 2), start=2):
            lower_bound = val
            upper_bound = val + 2
            segmented[(mask >= lower_bound) & (mask < upper_bound)] = i
        
        # Values > 22 and < 30: Last class (9)
        segmented[(mask > 22) & (mask < 30)] = 9
        
        # Value == 30: Also last class (9)
        segmented[mask == 30] = 9

        return segmented
    
    def return_pixelwise_weight_dw(self, dw_alpha):

        masks = None
        for idx, row in self.NewDf.iterrows():
            xcoord     = row['X'] 
            ycoord     = row['Y'] 
            label_path = row['LABEL_PATH'] 
            mask  = self._crop_gen(label_path, xcoord, ycoord) 
            mask  = np.swapaxes(mask, -1, 0)

            if masks is None: 
                masks = mask
            else: 
                masks = np.concatenate([masks, mask], axis = 0)

        reshaped_masks = np.reshape(masks, (masks.shape[0]*masks.shape[1]*masks.shape[2]))

        weights = TargetRelevance(reshaped_masks, alpha = dw_alpha).__call__(reshaped_masks)

        weights = np.reshape(weights, (masks.shape[0], masks.shape[1], masks.shape[2]))
        return weights  
    
    def _crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        if src.ndim == 2:
            src = np.expand_dims(src, axis = 0)
            src = np.expand_dims(src, axis = -1)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
    
    def patch_cultivar_matrix(self, cul_id):
        cultivar_matrix = np.full((1, self.wsize, self.wsize, 15), (1/cul_id)) 
        
        return cultivar_matrix
    
    def patch_rw_matrix(self, rw):
        rw_matrix = np.full((1, self.wsize, self.wsize, 15), (1/rw)) 
        
        return rw_matrix    
    
    def patch_sp_matrix(self, sp):
        sp_matrix = np.full((1, self.wsize, self.wsize, 15), (1/sp)) 
        return sp_matrix
        
    def patch_tid_matrix(self, tid):
        tid_matrix = np.full((1, self.wsize, self.wsize, 15), (1/tid))  
        
        return tid_matrix
    
    def add_input_within_bc_mean(self, bloks_mean):
        
        fill_matrix_bmean = np.full((1, self.wsize, self.wsize, 15), bloks_mean) / 30. 

        return fill_matrix_bmean
    
    def histogram_equalization_4d(self, image):
        """
        Apply histogram equalization to each channel of each timeseries frame in the image,
        ensuring each slice is an 8-bit single-channel image.

        Args:
        - image (numpy.ndarray): Input image array with values normalized to 0-255 and shape (C, H, W, T).

        Returns:
        - (numpy.ndarray): The histogram equalized image.
        """
        # Check if image dtype is uint8, convert if necessary
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Prepare the output array with the same shape
        eq_image = np.empty_like(image)

        # Iterate over each channel and timeseries
        for c in range(image.shape[0]):  # For each channel
            for t in range(image.shape[3]):  # For each time point
                # Apply histogram equalization to each slice (channel, :, :, timeseries)
                eq_image[c, :, :, t] = cv2.equalizeHist(image[c, :, :, t])

        return eq_image

    def _time_series_encoding(self, block_id):
        timeseries = None

        name_split = os.path.split(str(block_id))[-1]
        year       = name_split[-4:]

        if year == '2016': 
            days = [91, 95, 107, 116, 135, 136, 141, 150, 161, 166, 171, 176, 182, 195, 202]
        elif year =='2017':
            days = [90, 109, 119, 134, 140, 156, 169, 176, 179, 184, 190, 192, 195, 197, 202]
        elif year =='2018':
            days = [91, 105, 112, 115, 121, 131, 135, 142, 152, 155, 165, 175, 185, 191, 194]
        elif year =='2019':
            days = [91, 101, 112, 115, 121, 124, 131, 145, 152, 155, 164, 171, 181, 192, 202]
        
        for day in days: 
            this_week_matrix = np.full((1, self.wsize, self.wsize), 1 - np.sin(day/(365*np.pi))) 
            this_week_matrix = np.expand_dims(this_week_matrix, axis = -1)
            if timeseries is None:
                timeseries = this_week_matrix
            else:
                timeseries   = np.concatenate([timeseries, this_week_matrix], axis = -1)

        return timeseries  

def sns_inference_dataloader(batch_size: int, 
                         exp_name: str, 
                         keyword: str, 
                         new_avlue: float):

    root_data_dir = '/data2/hkaman/Data/'
    root_exp_dir = '/data2/hkaman/Projects/'

    exp_output_dir = root_exp_dir + 'ViT/EXPs/Sep/' + 'EXP_' + exp_name
    data_dir = root_data_dir + 'Livingston/data/10m/'
    
    dataset_test = InfDataCreator(
        data_dir, 
        exp_output_dir,  
        patch_size = 16, 
        in_channels = 8,
        keyword = keyword, 
        new_avlue = new_avlue
    )  

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                                shuffle=False, num_workers=8) 
    
    return data_loader_test

class InfDataCreator(object):
    def __init__(self, npy_dir, csv_dir,  
                                patch_size: int, 
                                in_channels: int, 
                                keyword: str, 
                                new_avlue: float
                                ):

        self.npy_dir = npy_dir
        self.csv_dir = csv_dir
        self.wsize = patch_size
        self.in_channels  = in_channels
        self.keyword = keyword
        self.new_avlue = new_avlue

        self.NewDf = pd.read_csv(os.path.join(self.csv_dir, 'coords') +'/test.csv', index_col=0)
        self.NewDf.reset_index(inplace = True, drop = True)

        self.stats_dict = np.load('/data2/hkaman/Data/Coords/met_stats.npz', allow_pickle=True)
        self.stats_dict = self.stats_dict['arr_0'].item() 

    def __getitem__(self, idx):

        xcoord = self.NewDf.loc[idx]['X'] 
        ycoord = self.NewDf.loc[idx]['Y'] 
        block_id = self.NewDf.loc[idx]['block']
        cultivar = self.NewDf.loc[idx]['cultivar']
        cultivar_id = self.NewDf.loc[idx]['cultivar_id']
        rw_id = self.NewDf.loc[idx]['row']
        sp_id = self.NewDf.loc[idx]['space']
        t_id = self.NewDf.loc[idx]['trellis_id']

        S2_path = self.NewDf.loc[idx]['IMG_PATH']
        S2 = self._crop_gen(S2_path, xcoord, ycoord) 
        S2 = np.swapaxes(S2, -1, 0)   
        S2 = self.histogram_equalization_4d(S2)
        S2 = S2 / 255.

        block_timeseries_encode = self._time_series_encoding(block_id)
        WithinBlockMean = self.NewDf.loc[idx]['win_block_mean']
        block_means = self.add_input_within_bc_mean(WithinBlockMean)

        S1_path = self.NewDf.loc[idx]['S1_PATH']
        S1 = self._crop_gen(S1_path, xcoord, ycoord) 
        S1 = np.swapaxes(S1, -1, 0)

        image = np.concatenate([S2, S1, block_timeseries_encode, block_means], axis = 0)
        image = torch.as_tensor(image, dtype=torch.float32)

        # MASK 
        label_path = self.NewDf.loc[idx]['LABEL_PATH']
        mask = self._crop_gen(label_path, xcoord, ycoord) 
        mask = np.swapaxes(mask, -1, 0)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        # YIELDZONE: 
        yz = self._YieldZoneMap(mask, num_classes= 9)
        yz = torch.as_tensor(yz, dtype=torch.float32)

        # Text
        text_path = self.NewDf.loc[idx]['TEXT_PATH']
        EmbText = self._load_text_file(text_path)
        EmbText = self._modify_value(EmbText, self.keyword, self.new_avlue)
        
        # Meteorological data
        met_path = self.NewDf.loc[idx]['MET_PATH']
        met = np.load(met_path)
        met = self._met_normalizer(met[..., 0], method= 'z-score')

        
        sample = {"image": image, 
                  "mask": mask, 
                  "met": met, 
                  "block": block_id, 
                  "cultivar": cultivar, 
                  "X": xcoord, "Y": ycoord, 
                  "EmbList": [cultivar_id, t_id, rw_id, sp_id], 
                  "EmbText": EmbText, 
                  "YZ": yz,} 
            
        return sample

    def __len__(self):
        return len(self.NewDf)
    
    def _load_text_file(self, file_path):
        # Extract the file extension to determine how to process it
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.pdf':
            pdf_loader = PdfReader(open(file_path, "rb"))
            file_text = ""
            for page_num in range(len(pdf_loader.pages)):
                pdf_page = pdf_loader.pages[page_num]
                if pdf_page.extract_text() is not None:
                    file_text += pdf_page.extract_text()
            return file_text
        
        elif file_extension.lower() == '.txt':
            with open(file_path, "r", encoding="utf-8") as file:
                file_text = file.read()
            return file_text
        
        else:
            raise ValueError("Unsupported file format: " + file_extension)

    def _modify_value(self, text, keyword, modified_value):
        if len(text) < 1000:
            return text  # Skip if text is too short

        # Specific pattern for 'electrical conductivity' to handle "at"
        if keyword == 'electrical conductivity':
            pattern = rf"{re.escape(keyword)}.*?at\s*([\d\.]+)"
        else:
            # General pattern for other keywords
            pattern = rf"\b{re.escape(keyword)}\b.*?([\d\.]+)\s*(?:[a-zA-Z%\/\(\)]*)"

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            original_value = float(match.group(1))  # Extract the current value
            new_value = modified_value
            
            # Modify the text based on the keyword
            if keyword == 'electrical conductivity':
                # For 'electrical conductivity', handle "at" in the replacement
                modified_text = re.sub(rf"({keyword}.*?at\s*){original_value}", f"\\1{new_value:.2f}", text, flags=re.IGNORECASE | re.DOTALL)
            else:
                # General replacement for other keywords
                modified_text = re.sub(rf"({keyword})\s*{original_value}", f"\\1 {new_value:.2f}", text, flags=re.IGNORECASE)

            return modified_text
        else:
            raise ValueError(f"Keyword '{keyword}' not found in the text.")
        
        # if len(text) < 100:
        #     return text  
            
        # if keyword == 'electrical conductivity':
        #     pattern = rf"{re.escape(keyword)}.*?at\s*([\d\.]+)" 
        # else:
        #     pattern = rf"\b{re.escape(keyword)}\b.*?([\d\.]+)\s*(?:[a-zA-Z%\/\(\)]*)"
    
        # match = re.search(pattern, text, re.IGNORECASE | re.DOTALL) 
    
        # # pattern = rf"\b{re.escape(keyword)}\b\s*([\d\.]+)"
        # # match = re.search(pattern, text, re.IGNORECASE)

        # if match:
        #     original_value = float(match.group(1))  
        #     new_value = modified_value
        #     modified_text = re.sub(rf"({keyword})\s*{original_value}", f"\\1 {new_value:.2f}", text, flags=re.IGNORECASE)
        #     return modified_text
        # else:
        #     raise ValueError(f"Keyword '{keyword}' not found in the text.")





    def _met_normalizer(self, arr, method='z-score'):
        """
        Normalize the meteorological dataset using either min-max normalization or z-score normalization.

        Parameters
        ----------
        arr : np.ndarray
            The input array to normalize, with the shape (channels, ...).
        stats_dict : dict
            The dictionary containing the statistics (mean, std, min, max) for each variable.
        method : str
            The normalization method, either 'min-max' or 'z-score'.
            
        Returns
        -------
        np.ndarray
            The normalized array.
        """
        for channel in range(1, 4):
            channel_key = {1: 'tmin', 2: 'tmax', 3: 'vp'}[channel]
            
            if method == 'z-score':
                mean_values = np.array(self.stats_dict[channel_key]['mean'])
                std_values = np.array(self.stats_dict[channel_key]['std'])

                for w in range(15):
                    arr[channel, ...] = (arr[channel, ...] - mean_values[w]) / std_values[w]

            elif method == 'min-max':
                min_values = np.array(self.stats_dict[channel_key]['min'])
                max_values = np.array(self.stats_dict[channel_key]['max'])

                for w in range(15):
                    arr[channel, ...] = (arr[channel, ...] - min_values[w]) / (max_values[w] - min_values[w])

        return arr

    def return_yield_zone(self, mask):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented = np.zeros_like(mask)
        # Class 1: Pixel values <= 9
        segmented[mask < EXTREME_LOWER_THRESHOLD] = 1
        # Class 2: Pixel values > 9 and < 22
        segmented[(mask >= EXTREME_LOWER_THRESHOLD) & (mask < EXTREME_UPPER_THRESHOLD)] = 2
        # Class 3: Pixel values >= 22
        segmented[mask >= EXTREME_UPPER_THRESHOLD] = 3

        return segmented
    
    def _YieldZoneMap(self, mask, num_classes: int):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented = np.zeros_like(mask)
        # min_point = 8
        # max_point = 22
        
        if num_classes == 3: 
            # Initialize an empty array with the same shape as the image for the segmented output
            segmented = np.zeros_like(mask)
            # Class 1: Pixel values <= 9
            segmented[mask < EXTREME_LOWER_THRESHOLD] = 1
            # Class 2: Pixel values > 9 and < 22
            segmented[(mask >= EXTREME_LOWER_THRESHOLD) & (mask < EXTREME_UPPER_THRESHOLD)] = 2
            # Class 3: Pixel values >= 22
            segmented[mask >= EXTREME_UPPER_THRESHOLD] = 3

        elif num_classes == 9:
            # Values < 8: Class 1
            segmented[mask < 8] = 1
            
            # Values between 8 and 22: Classes 2 to 8 (7 intervals of 2)
            for i, val in enumerate(range(EXTREME_LOWER_THRESHOLD, EXTREME_UPPER_THRESHOLD, 2), start=2):
                lower_bound = val
                upper_bound = val + 2
                segmented[(mask >= lower_bound) & (mask < upper_bound)] = i
            
            # Values > 22 and < 30: Last class (9)
            segmented[(mask > EXTREME_UPPER_THRESHOLD) & (mask < 30)] = 9
    
            # Value == 30: Also last class (9)
            segmented[mask == 30] = 9

        elif num_classes == 15:
            for i in range(15):
                lower_bound = i * 2
                upper_bound = (i + 1) * 2
                segmented[(mask >= lower_bound) & (mask < upper_bound)] = i + 1
            
            # Special case for the upper boundary of the last class to include the value 30
            segmented[mask == 30] = 15

        return segmented
    
    def return_yield_zone_9_classes(self, mask):
        # Initialize an empty array with the same shape as the image for the segmented output
        segmented = np.zeros_like(mask)
        
        # Values < 8: Class 1
        segmented[mask < 8] = 1
        
        # Values between 8 and 22: Classes 2 to 8 (7 intervals of 2)
        for i, val in enumerate(range(8, 22, 2), start=2):
            lower_bound = val
            upper_bound = val + 2
            segmented[(mask >= lower_bound) & (mask < upper_bound)] = i
        
        # Values > 22 and < 30: Last class (9)
        segmented[(mask > 22) & (mask < 30)] = 9
        
        # Value == 30: Also last class (9)
        segmented[mask == 30] = 9

        return segmented
    
    def return_pixelwise_weight_dw(self, dw_alpha):

        masks = None
        for idx, row in self.NewDf.iterrows():
            xcoord     = row['X'] 
            ycoord     = row['Y'] 
            label_path = row['LABEL_PATH'] 
            mask  = self._crop_gen(label_path, xcoord, ycoord) 
            mask  = np.swapaxes(mask, -1, 0)

            if masks is None: 
                masks = mask
            else: 
                masks = np.concatenate([masks, mask], axis = 0)

        reshaped_masks = np.reshape(masks, (masks.shape[0]*masks.shape[1]*masks.shape[2]))

        weights = TargetRelevance(reshaped_masks, alpha = dw_alpha).__call__(reshaped_masks)

        weights = np.reshape(weights, (masks.shape[0], masks.shape[1], masks.shape[2]))
        return weights  
    
    def _crop_gen(self, src, xcoord, ycoord):
        src = np.load(src, allow_pickle=True)
        if src.ndim == 2:
            src = np.expand_dims(src, axis = 0)
            src = np.expand_dims(src, axis = -1)
        crop_src = src[:, xcoord:xcoord + self.wsize, ycoord:ycoord + self.wsize, :]
        return crop_src 
    
    def patch_cultivar_matrix(self, cul_id):
        cultivar_matrix = np.full((1, self.wsize, self.wsize, 15), (1/cul_id)) 
        
        return cultivar_matrix
    
    def patch_rw_matrix(self, rw):
        rw_matrix = np.full((1, self.wsize, self.wsize, 15), (1/rw)) 
        
        return rw_matrix    
    
    def patch_sp_matrix(self, sp):
        sp_matrix = np.full((1, self.wsize, self.wsize, 15), (1/sp)) 
        return sp_matrix
        
    def patch_tid_matrix(self, tid):
        tid_matrix = np.full((1, self.wsize, self.wsize, 15), (1/tid))  
        
        return tid_matrix
    
    def add_input_within_bc_mean(self, bloks_mean):
        
        fill_matrix_bmean = np.full((1, self.wsize, self.wsize, 15), bloks_mean) / 30. 

        return fill_matrix_bmean
    
    def histogram_equalization_4d(self, image):
        """
        Apply histogram equalization to each channel of each timeseries frame in the image,
        ensuring each slice is an 8-bit single-channel image.

        Args:
        - image (numpy.ndarray): Input image array with values normalized to 0-255 and shape (C, H, W, T).

        Returns:
        - (numpy.ndarray): The histogram equalized image.
        """
        # Check if image dtype is uint8, convert if necessary
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Prepare the output array with the same shape
        eq_image = np.empty_like(image)

        # Iterate over each channel and timeseries
        for c in range(image.shape[0]):  # For each channel
            for t in range(image.shape[3]):  # For each time point
                # Apply histogram equalization to each slice (channel, :, :, timeseries)
                eq_image[c, :, :, t] = cv2.equalizeHist(image[c, :, :, t])

        return eq_image

    def _time_series_encoding(self, block_id):
        timeseries = None

        name_split = os.path.split(str(block_id))[-1]
        year       = name_split[-4:]

        if year == '2016': 
            days = [91, 95, 107, 116, 135, 136, 141, 150, 161, 166, 171, 176, 182, 195, 202]
        elif year =='2017':
            days = [90, 109, 119, 134, 140, 156, 169, 176, 179, 184, 190, 192, 195, 197, 202]
        elif year =='2018':
            days = [91, 105, 112, 115, 121, 131, 135, 142, 152, 155, 165, 175, 185, 191, 194]
        elif year =='2019':
            days = [91, 101, 112, 115, 121, 124, 131, 145, 152, 155, 164, 171, 181, 192, 202]
        
        for day in days: 
            this_week_matrix = np.full((1, self.wsize, self.wsize), 1 - np.sin(day/(365*np.pi))) 
            this_week_matrix = np.expand_dims(this_week_matrix, axis = -1)
            if timeseries is None:
                timeseries = this_week_matrix
            else:
                timeseries   = np.concatenate([timeseries, this_week_matrix], axis = -1)

        return timeseries  

























































from KDEpy import FFTKDE
from sklearn.preprocessing import MinMaxScaler

def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.
    From https://stackoverflow.com/a/41856629'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl

class TargetRelevance():

    def __init__(self, y, alpha=1.0):
        self.alpha = alpha
       #print('TargetRelevance alpha:', self.alpha)

        silverman_bandwidth = 1.06*np.std(y)*np.power(len(y), (-1.0/5.0))

        #print('Using Silverman Bandwidth', silverman_bandwidth)
        best_bandwidth = silverman_bandwidth

        self.kernel = FFTKDE(bw=best_bandwidth).fit(y, weights=None)

        x, y_dens_grid = self.kernel.evaluate(1024)  # Default precision is 1024
        self.x = x
        
        # Min-Max Scale to 0-1 since pdf's can actually exceed 1
        # See: https://stats.stackexchange.com/questions/5819/kernel-density-estimate-takes-values-larger-than-1
        self.y_dens_grid = MinMaxScaler().fit_transform(y_dens_grid.reshape(-1, 1)).flatten()

        self.y_dens = np.vectorize(self.get_density)(y)

        self.eps = 1e-6
        w_star = np.maximum(1 - self.alpha * self.y_dens, self.eps)
        self.mean_w_star = np.mean(w_star)
        self.relevances = w_star / self.mean_w_star

    def get_density(self, y):
        idx = bisection(self.x, y)
        try:
            dens = self.y_dens_grid[idx]
        except IndexError:
            if idx <= -1:
                idx = 0
            elif idx >= len(self.x):
                idx = len(self.x) - 1
            dens = self.y_dens_grid[idx]
        return dens

    #@functools.lru_cache(maxsize=100000)
    def eval_single(self, y):
        dens = self.get_density(y)
        return np.maximum(1 - self.alpha * dens, self.eps) / self.mean_w_star

    def eval(self, y):
        ys = y.flatten().tolist()
        rels = np.array(list(map(self.eval_single, ys)))[:, None]
        return rels

    def __call__(self, y):
        return self.eval(y)