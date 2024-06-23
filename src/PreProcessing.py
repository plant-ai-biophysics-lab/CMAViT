import os
import os.path as path
import numpy as np
from glob import glob
import rasterio 
from rasterio.plot import show, show_hist
from rasterio.mask import mask
from rasterio.coords import BoundingBox
from rasterio import windows
from rasterio import warp
from rasterio.merge import merge
import cv2
import xarray as xr
from skimage.restoration import denoise_nl_means, estimate_sigma
import rioxarray


root_dir = '/data2/hkaman/Data/Livingston/' 
def sentinel1_normalize_band(band: xr.DataArray) -> xr.DataArray:
    """
    Normalize a band of data between 0 and 1 by computing min and max values from the data.
    
    Parameters
    ----------
    band : xr.DataArray
        The input band to normalize.
        
    Returns
    -------
    xr.DataArray
        The normalized band.
    """
    min_val = band.min().item()
    max_val = band.max().item()
    return (band - min_val) / (max_val - min_val)

# def sentinel1_denoising(
#     dataset: xr.DataArray,
#     patch_size: int = 5,
#     patch_distance: int = 6,
# ) -> xr.DataArray:
#     """
#     Image denoising using non-local means filter for Sentinel-1 data.
#     Non-local means filters identify similar image patches
#     in a dataset and use those patches to optimally suppress
#     noise without sacrificing image resolution.

#     Parameters
#     ----------
#     dataset: xr.DataArray
#         The input dataset to denoise.
#     patch_size: int
#         Patch size for non-local means filter.
#     patch_distance: int
#         Patch distance for non-local means filter.

#     Returns
#     -------
#     xr.DataArray
#         The denoised dataset.
#     """
#     coords = ('y', 'x')

#     def _denoise(val):
#         sigma = estimate_sigma(val)
#         print(sigma)
#         val = denoise_nl_means(
#             val,
#             h=0.2 * sigma,
#             sigma=sigma,
#             fast_mode=True,
#             patch_size=patch_size,
#             patch_distance=patch_distance,
#         )
#         return val

#     denoised_bands = []
#     for band_idx in dataset.band.values:
#         band_data = dataset.sel(band=band_idx)
#         denoised_band = xr.apply_ufunc(
#             _denoise,
#             band_data,
#             input_core_dims=[coords],
#             output_core_dims=[coords],
#             vectorize=True,
#         )
#         denoised_bands.append(denoised_band)

#     denoised_dataset = xr.concat(denoised_bands, dim='band')
#     denoised_dataset['band'] = dataset.band

#     return denoised_dataset



def sentinel1_denoising(
    dataset: xr.DataArray,
    patch_size: int = 5,
    patch_distance: int = 6,
) -> xr.DataArray:
    """
    Image denoising using non-local means filter for Sentinel-1 data.
    Non-local means filters identify similar image patches
    in a dataset and use those patches to optimally suppress
    noise without sacrificing image resolution.

    Parameters
    ----------
    dataset: xr.DataArray
        The input dataset to denoise.
    patch_size: int
        Patch size for non-local means filter.
    patch_distance: int
        Patch distance for non-local means filter.

    Returns
    -------
    xr.DataArray
        The denoised dataset, or original dataset if denoising fails.
    """
    coords = ('y', 'x')

    def _denoise(val):
        sigma = estimate_sigma(val)
        val = denoise_nl_means(
            val,
            h=0.2 * sigma,
            sigma=sigma,
            fast_mode=True,
            patch_size=patch_size,
            patch_distance=patch_distance,
        )
        return val

    try:
        denoised_bands = []
        for band_idx in dataset.band.values:
            band_data = dataset.sel(band=band_idx)
            denoised_band = xr.apply_ufunc(
                _denoise,
                band_data,
                input_core_dims=[coords],
                output_core_dims=[coords],
                vectorize=True,
            )
            denoised_bands.append(denoised_band)

        denoised_dataset = xr.concat(denoised_bands, dim='band')
        denoised_dataset['band'] = dataset.band
        return denoised_dataset

    except Exception as e:
        print(f"An error occurred during denoising: {e}")
        return dataset  # Return the original dataset if an error occurs


def _power_to_db(dataset: xr.Dataset) -> xr.Dataset:
    """
    Function to convert SAR units from power to dB
    """
    return 10 * np.log10(dataset)

def process_sentinel1_and_save_tifs(input_dir: str, output_dir: str):
    """
    Process TIFF files: denoise and normalize, then save to the output directory.
    
    Parameters
    ----------
    input_dir : str
        The directory containing the input TIFF files.
    output_dir : str
        The directory to save the processed TIFF files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.tif'):
            output_filepath = os.path.join(output_dir, filename)
            if not os.path.exists(output_filepath):

                filepath = os.path.join(input_dir, filename)
                dataset = rioxarray.open_rasterio(filepath)

                dinoised_dataset = sentinel1_denoising(dataset)

                vv_band = dinoised_dataset.sel(band=1)
                vh_band = dinoised_dataset.sel(band=2)

                vv_band_normalized = sentinel1_normalize_band(vv_band)
                vh_band_normalized = sentinel1_normalize_band(vh_band)

                # Combine the normalized bands back into a single dataset
                normalized_dataset = xr.concat([vv_band_normalized, vh_band_normalized], dim='band')
                normalized_dataset['band'] = [1, 2]

                # Save the processed dataset
                normalized_dataset.rio.to_raster(output_filepath)   
                print(f"Processed and saved: {output_filepath}")

def sentinel1_block_npy_gen(data_dir: str, 
                            img_save_dir:str, 
                            year:str):
    LIV_year = 'LIV' + year

    image_dir = os.path.join(data_dir + LIV_year + '/image/Sentinel1_dinoised/')
    image_names = os.listdir(image_dir)
    image_names.sort()

    label_dir = os.path.join(data_dir + LIV_year + '/yield/')
    label_names = os.listdir(label_dir)
    label_names.sort()

    if year == '2016':
        # because the format of 2016 is not tif!
        format = ".asc"
    else:
        format = ".tif"


    for name in label_names: 
        if name.endswith(format):
            label_file_name = label_dir + '/' + name  
            label_src = rasterio.open(label_file_name)
            label_data = label_src.read()
            label_data = np.moveaxis(label_data, 0, 2)
            # convert all the non values including the background to -1 
            label_data[label_data<0] = -1 
            # adding one more dimension to be able for concatenation 
            
            #print(label_data.shape)

            h = label_src.height # hight of label image
            w = label_src.width  # width of label image 
            slice_ = (slice(0, h), slice(0, w))
            window_slice = windows.Window.from_slices(*slice_)
            # Window to list of index (row,col) 
            pol_index = to_index(window_slice)

            # Convert list of index (row,col) to list of coordinates of lat and long 
            pol = [list(label_src.transform*p) for p in reverse_coordinates(pol_index)]
            
            roi_polygon_src_coords = warp.transform_geom(label_src.crs,
                                            {'init': 'epsg:32610'},                                          
                                            {"type": "Polygon",
                                            "coordinates": [pol]})
            
            # aggregate the yeild map to 10m resolution 
            label_data = cv2.resize(label_data, None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_LINEAR)
            h = label_data.shape[0]
            w = label_data.shape[1]

            label_data = np.expand_dims(label_data, axis = 0)
            label_data = np.expand_dims(label_data, axis = -1)

            timeseries_image = None
            for img in image_names:
                img_name =  image_dir + '/'+ img

                _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                image_cropped = np.moveaxis(image_cropped, 0, 2)
                image_cropped = np.nan_to_num(image_cropped)

                image_cropped = image_cropped[0:h, 0:w,:]
                image_cropped_ex = np.expand_dims(image_cropped, axis = 0)

                if timeseries_image is None:
                    timeseries_image = image_cropped_ex
                else:
                    timeseries_image = np.concatenate((timeseries_image, image_cropped_ex), axis = 0)

            #print(f"{label_data.shape}|{timeseries_image.shape}")

            name_split = os.path.split(name)[-1]
            name_root = name_split.replace(name_split[-4:], '')
            new_img_name = name_root + '_img.npy'
            img_npy_name = img_save_dir + new_img_name
            _ = saveList(timeseries_image, img_npy_name)

def sentinel_image_stretch(image): 
    # normalize into 0-255
    red = image[0,:,:]
    red_n = ((255*red/np.max(red))).astype(np.uint8)
    red_n = np.expand_dims(red_n, axis = 0)

    green = image[1,:,:]
    green_n = ((255*green/np.max(green))).astype(np.uint8)
    green_n = np.expand_dims(green_n, axis = 0)


    blue = image[2,:,:]
    blue_n = ((255*blue/np.max(blue))).astype(np.uint8)
    blue_n = np.expand_dims(blue_n, axis = 0)

    nir = image[3,:,:]
    nir_n = ((255*nir/np.max(nir))).astype(np.uint8)
    nir_n = np.expand_dims(nir_n, axis = 0) 

    image_norm = np.concatenate([red_n, green_n, blue_n, nir_n], axis = 0)

    return image_norm

def img_crop(img, polygon):
    with rasterio.open(img) as inimg:
        in_cropped, out_transform_in = mask(inimg,
        [polygon],crop=True)
        in_cropped_meta = inimg.meta.copy()
        in_cropped_meta.update({"driver": "GTiff",
            "height": in_cropped.shape[1],
            "width": in_cropped.shape[2], 
            "transform": out_transform_in})
    return in_cropped_meta, in_cropped

def reverse_coordinates(pol):
    """
    Reverse the coordinates in pol
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    Returns [[y1,x1],[y2,x2],...,[yN,xN]]
    """
    #return [list(f[-1::-1]) for f in pol]
    list = []
    for f in pol:
        f2 = f[-1::-1]
        list.append(f2)
    return list

def list_to_coord (list, src):
    coord = []
    for p in list:
        cor = src.transform*p
        coord.append(cor)

    return coord

def to_index(wind_):
    """
    Generates a list of index (row,col): [[row1,col1],[row2,col2],[row3,col3],[row4,col4],[row1,col1]]
    """
    return [[wind_.row_off,wind_.col_off],
            [wind_.row_off,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off+wind_.width],
            [wind_.row_off+wind_.height,wind_.col_off],
            [wind_.row_off,wind_.col_off]]

def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
            [bbox[2],bbox[1]],
            [bbox[2],bbox[3]],
            [bbox[0],bbox[3]],
            [bbox[0],bbox[1]]]

def pol_to_np(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    return np.array([list(l) for l in pol])

def pol_to_bounding_box(pol):
    """
    Receives list of coordinates: [[x1,y1],[x2,y2],...,[xN,yN]]
    """
    arr = pol_to_np(pol)
    return BoundingBox(np.min(arr[:,0]),
                    np.min(arr[:,1]),
                    np.max(arr[:,0]),
                    np.max(arr[:,1]))   

def block_npy_gen(data_dir, img_save_dir, label_save_dir, year = None, spatial_res = None):
    LIV_year = 'LIV' + year

    image_dir = os.path.join(data_dir + LIV_year+ '/image/Sentinel2/')
    image_names = os.listdir(image_dir)
    image_names.sort()

    label_dir = os.path.join(data_dir+ LIV_year+'/yield/')
    label_names = os.listdir(label_dir)
    label_names.sort()

    output_list = []

    if year == '2016':
        # because the format of 2016 is not tif!
        format = ".asc"
    else:
        format = ".tif"


    for name in label_names: 
        if name.endswith(format):
            label_file_name = label_dir + '/' + name  
            label_src = rasterio.open(label_file_name)
            label_data = label_src.read()
            label_data = np.moveaxis(label_data, 0, 2)
            # convert all the non values including the background to -1 
            label_data[label_data<0] = -1 
            # adding one more dimension to be able for concatenation 
            
            #print(label_data.shape)

            h = label_src.height # hight of label image
            w = label_src.width  # width of label image 
            slice_ = (slice(0, h), slice(0, w))
            window_slice = windows.Window.from_slices(*slice_)
            # Window to list of index (row,col) 
            pol_index = to_index(window_slice)

            # Convert list of index (row,col) to list of coordinates of lat and long 
            pol = [list(label_src.transform*p) for p in reverse_coordinates(pol_index)]
            
            roi_polygon_src_coords = warp.transform_geom(label_src.crs,
                                            {'init': 'epsg:32610'},                                          
                                            {"type": "Polygon",
                                            "coordinates": [pol]})
            

            if spatial_res == 1: 
                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)

                timeseries_image = None

                for img in image_names:
                    img_name =  image_dir + '/'+ img
                    # crop the sentinel image based on the corner of block shapefile corrdinates
                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    # sentinel image normalization! 
                    image_cropped = sentinel_image_stretch(image_cropped) 
                    # change the order of channels.
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    # convert nan to zero
                    image_cropped = np.nan_to_num(image_cropped)
                    # adding one more dimension
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
   
                    
                    up10x_img = cv2.resize(image_cropped, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)
                    up10x_img = up10x_img[0:h, 0:w,:]
                    up10x_img_ex = np.expand_dims(up10x_img, axis = 0)
                    if timeseries_image is None:
                        timeseries_image = up10x_img_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, up10x_img_ex), axis = 0) 
                #print(np.sum(np.array(up10x_img) == 0)

            elif spatial_res == 10: 
                # aggregate the yeild map to 10m resolution 
                label_data = cv2.resize(label_data, None, fx = 0.1, fy = 0.1, interpolation = cv2.INTER_LINEAR)
                h = label_data.shape[0]
                w = label_data.shape[1]

                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)

                timeseries_image = None
                for img in image_names:
                    img_name =  image_dir + '/'+ img

                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    #print(image_cropped.shape)
                    image_cropped = sentinel_image_stretch(image_cropped)
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    #print(image_cropped.shape)
                    image_cropped = np.nan_to_num(image_cropped)
                    #print(image_cropped.shape)
                    image_cropped = image_cropped[0:h, 0:w,:]
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
                    #print(image_cropped_ex.shape)
                    if timeseries_image is None:
                        timeseries_image = image_cropped_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, image_cropped_ex), axis = 0)

            elif spatial_res == 20: 

                label_data = cv2.resize(label_data, None, fx = 0.05, fy = 0.05, interpolation = cv2.INTER_LINEAR)
                h = label_data.shape[0]
                w = label_data.shape[1]

                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)


                timeseries_image = None
                for img in image_names:
                    img_name =  image_dir + '/'+ img

                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    #print(image_cropped.shape)
                    image_cropped = sentinel_image_stretch(image_cropped)
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    #print(image_cropped.shape)

                    image_cropped = np.nan_to_num(image_cropped)
                    #print(image_cropped.shape)
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
                    #print(image_cropped_ex.shape)


                    up10x_img = cv2.resize(image_cropped, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
                    up10x_img = up10x_img[0:h, 0:w,:]
                    up10x_img_ex = np.expand_dims(up10x_img, axis = 0)


                    if timeseries_image is None:
                        timeseries_image = up10x_img_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, up10x_img_ex), axis = 0) 

            elif spatial_res == 40: 

                label_data = cv2.resize(label_data, None, fx = 0.025, fy = 0.025, interpolation = cv2.INTER_LINEAR)
                h = label_data.shape[0]
                w = label_data.shape[1]

                label_data = np.expand_dims(label_data, axis = 0)
                label_data = np.expand_dims(label_data, axis = -1)
                
                
                timeseries_image = None
                img_concate = None
                for img in image_names:
                    img_name =  image_dir + '/'+ img

                    _, image_cropped  = img_crop(img_name, roi_polygon_src_coords)
                    #print(image_cropped.shape)
                    image_cropped = sentinel_image_stretch(image_cropped)
                    image_cropped = np.moveaxis(image_cropped, 0, 2)
                    #print(image_cropped.shape)

                    image_cropped = np.nan_to_num(image_cropped)
                    #print(image_cropped.shape)
                    image_cropped_ex = np.expand_dims(image_cropped, axis = 0)
                    #print(image_cropped_ex.shape)

                    up10x_img = cv2.resize(image_cropped, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_CUBIC)
                    up10x_img = up10x_img[0:h, 0:w,:]
                    up10x_img_ex = np.expand_dims(up10x_img, axis = 0)
                    if timeseries_image is None:
                        timeseries_image = up10x_img_ex
                    else:
                        timeseries_image = np.concatenate((timeseries_image, up10x_img_ex), axis = 0) 




            #print(f"{label_data.shape}|{timeseries_image.shape}")

            name_split = os.path.split(name)[-1]
            name_root = name_split.replace(name_split[-4:], '')
            new_img_name = name_root + '_img.npy'
            img_npy_name = img_save_dir + new_img_name
            _ = saveList(timeseries_image, img_npy_name)

            new_label_name = name_root + '_label.npy'
            label_npy_name = label_save_dir + new_label_name
            _ = saveList(label_data, label_npy_name) 

def tranfer_file(input_list, filter_list, src_dir, dst_dir):
    for f in input_list:
        
        if f in filter_list:
            src = src_dir+f
            dst = dst_dir+f
            shutil.move(src,dst)
            
def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename, myList)
    #print("Saved successfully!")

def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename, allow_pickle=True)
    return tempNumpyArray.tolist()
