import bitarray, os
from bitarray import util as butil

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
from tensorflow.keras.backend import bias_add, constant   

np.random.seed(724)

# Path to the openslide dll
OPENSLIDE_PATH = 'dependencies/openslide_win64_20230414_dlls'
# Importing openslide
if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def RGB2HSD(X): # Hue Saturation Density
    '''
    Function to convert RGB to HSD
    from https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/ops.py
    Args:
        X: RGB image
    Returns:
        X_HSD: HSD image
    '''
    eps = np.finfo(float).eps # Epsilon
    X[np.where(X==0.0)] = eps # Changing zeros with epsilon
    OD = -np.log(X / 1.0) # It seems to be calculating the Optical Density
    D  = np.mean(OD,3) # Getting density?
    D[np.where(D==0.0)] = eps # Changing zero densitites with epsilon
    cx = OD[:,:,:,0] / (D) - 1.0 
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    D = np.expand_dims(D,3) # Hue?
    cx = np.expand_dims(cx,3) # Saturation
    cy = np.expand_dims(cy,3) # Density?
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD

def clean_thumbnail(thumbnail):
    '''
    Function to clean thumbnail
    Args:
        thumbnail: thumbnail image
    Returns:
        wthumbnail: cleaned thumbnail image
    '''
    # thumbnail array
    thumbnail_arr = np.asarray(thumbnail)
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    # Remove pen marking here
    # We are skipping this
    # This  section sets regoins with white spectrum as the backgroud regoin
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD(np.array([wthumbnail.astype('float32')/255.]))[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    # return writable thumbnail
    return wthumbnail

def get_patches(slide, tissue_mask):
    '''
    Function to get patches
    Args:
        slide: slide object
        tissue_mask: tissue mask
    Returns:
        patches: list of patches
    '''
    # getting slide dimensions and objective power
    w, h = slide.dimensions
    objective_power = int(slide.properties['openslide.objective-power'])
    # at 20x its 1000x1000
    patch_size = (objective_power/20.)*1000
    # getting mask ratios
    mask_hratio = (tissue_mask.shape[0]/h)*patch_size
    mask_wratio = (tissue_mask.shape[1]/w)*patch_size
    # iterating over patches
    patches = []
    for i, hi in enumerate(range(0, h, int(patch_size))):
        # iterating over patches
        _patches = []
        for j, wi in enumerate(range(0, w, int(patch_size))):
            # check if patch contains 70% tissue area
            mi = int(i*mask_hratio)
            mj = int(j*mask_wratio)
            # get patch mask
            patch_mask = tissue_mask[mi:mi+int(mask_hratio), mj:mj+int(mask_wratio)]
            # get tissue coverage
            tissue_coverage = np.count_nonzero(patch_mask)/patch_mask.size
            # Add patch to list
            _patches.append({'loc': [i, j], 'wsi_loc': [int(hi), int(wi)], 'tissue_coverage': tissue_coverage})
        # Add patches to list
        patches.append(_patches)
    # return patches
    return patches
        

def get_flat_pathces(slide, patches, tissue_threshold):  
    '''
    Function to get flat patches
    Args:
        slide: slide object
        patches: list of patches
        tissue_threshold: threshold for tissue coverage
    Returns:
        flat_patches: list of flat patches
    '''
    # Converting patches to flat patches
    flat_patches = np.ravel(patches)
    # Getting the objective power
    objective_power = int(slide.properties['openslide.objective-power'])
    # Iterating over patches
    for patch in tqdm(flat_patches):
        # ignore patches with less tissue coverage
        if patch['tissue_coverage'] < tissue_threshold:
            continue
        # this loc is at the objective power
        h, w = patch['wsi_loc']
        # we will go one level lower, i.e. (objective power / 4)
        # we still need patches at 5x of size 250x250
        # this logic can be modified and may not work properly for images of lower objective power < 20 or greater than 40
        patch_size_5x = int(((objective_power / 4)/5)*250.)
        # read the patch
        patch_region = slide.read_region((w, h), 1, (patch_size_5x, patch_size_5x)).convert('RGB')
        # resize to 250x250
        if patch_region.size[0] != 250:
            patch_region = patch_region.resize((250, 250))
        # convert to numpy array
        histogram = (np.array(patch_region)/255.).reshape((250*250, 3)).mean(axis=0)
        patch['rgb_histogram'] = histogram  
    # return flat patches
    return flat_patches
        
def get_mosaic(flat_patches, kmeans_clusters, tissue_threshold, percentage_selected):
    '''
    Function to get mosaic from flat patches
    Args:
        flat_patches: list of patches
        kmeans_clusters: number of clusters for kmeans
        tissue_threshold: threshold for tissue coverage
        percentage_selected: percentage of patches to be selected from each cluster
    Returns:
        mosaic: list of patches to be used for mosaic
    '''
    # select patches with tissue coverage greater than threshold
    selected_patches_flags = [patch['tissue_coverage'] >= tissue_threshold for patch in flat_patches]
    selected_patches = flat_patches[selected_patches_flags]
    # Finding the number of patches as minimum of set kmeans_clusters and number of selected patches
    kmeans_clusters = min(kmeans_clusters, len(selected_patches))
    # run kmeans on rgb histogram
    kmeans = KMeans(n_clusters = kmeans_clusters, random_state=724)
    # get rgb histogram
    features = np.array([entry['rgb_histogram'] for entry in selected_patches])
    # fit kmeans
    kmeans.fit(features)
    # initialize mosaic
    mosaic = []
    # iterate over clusters
    for i in range(kmeans_clusters):
        # select patches from each cluster
        cluster_patches = selected_patches[kmeans.labels_ == i]
        # find number of patches to be selected from this cluster
        n_selected = max(1, int(len(cluster_patches)*percentage_selected/100.))
        # initialize kmeans
        km = KMeans(n_clusters=n_selected, random_state=724)
        # get location features
        loc_features = [patch['wsi_loc'] for patch in cluster_patches]
        # fit kmeans
        ds = km.fit_transform(loc_features)
        # initialize selected idx
        c_selected_idx = []
        # iterate over selected patches
        for idx in range(n_selected):
            # sort idx based on distance
            sorted_idx = np.argsort(ds[:, idx])
            # iterate over sorted idx
            for sidx in sorted_idx:
                if sidx not in c_selected_idx:
                    # add idx to selected idx if not already selected
                    c_selected_idx.append(sidx)
                    mosaic.append(cluster_patches[sidx])
                    break
    # return mosaic       
    return mosaic
        
def preprocessing_fn_densenet121(inp, sz=(1000, 1000)):
    '''
    Function to preprocess the input image for densenet121
    Args:
        inp: Input image
        sz: Size of the image
    Returns:
        out: Preprocessed image
    '''
    # cast to float
    out = tf.cast(inp, 'float') / 255.
    # resize
    out = tf.cond(tf.equal(tf.shape(inp)[1], sz[0]), 
                lambda: out, lambda: tf.image.resize(out, sz))
    # normalize
    mean = tf.reshape((0.485, 0.456, 0.406), [1, 1, 1, 3])
    std = tf.reshape((0.229, 0.224, 0.225), [1, 1, 1, 3])
    out = out - mean
    out = out / std
    # Return the output
    return out

def preprocessing_fn_kimianet(input_batch, network_input_patch_width):
    '''
    Function to preprocess the input batch for KimiaNet
    Args:
        input_batch: batch of images
        network_input_patch_width: width of the input patch
    Returns:
        standardized_input_batch: standardized input batch
    '''
    # get the original input size
    org_input_size = tf.shape(input_batch)[1]
    # standardization
    scaled_input_batch = tf.cast(input_batch, 'float') / 255.
    # resizing the patches if necessary
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width),
                                lambda: scaled_input_batch, 
                                lambda: tf.image.resize(scaled_input_batch, 
                                                        (network_input_patch_width, network_input_patch_width)))
    # normalization, this is equal to tf.keras.applications.densenet.preprocess_input()---------------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_format = "channels_last"
    mean_tensor = constant(-np.array(mean))
    standardized_input_batch = bias_add(resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= std
    return standardized_input_batch

def get_dn121_model():
    '''
    Function to get the DenseNet121 model
    Returns:
        seq_model: Sequential model with the DenseNet121 model
    '''
    # get the model
    model = tf.keras.applications.densenet.DenseNet121(input_shape=(1000, 1000, 3),\
                                                       include_top=False,\
                                                       pooling='avg')
    # add the preprocessing layer
    seq_model = tf.keras.models.Sequential([tf.keras.layers.Lambda(preprocessing_fn_densenet121,\
                                                   input_shape=(None, None, 3),\
                                                   dtype=tf.uint8)])
    # add the model
    seq_model.add(model)
    # return the model
    return seq_model

def get_kimianet_model(network_input_patch_width, weights_address='dependencies/kimianet/KimiaNetKerasWeights.h5'):
    '''
    Function to get the KimiaNet model
    Args:
        network_input_patch_width: width of the input patch
        weights_address: address of the weights
    Returns:
        kn_feature_extractor_seq: Sequential model with the KimiaNet model
    '''
    dnx = DenseNet121(include_top=False, weights=weights_address, 
                      input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')

    kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
    
    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn_kimianet, 
                                                  arguments={'network_input_patch_width': network_input_patch_width}, 
                                   input_shape=(None, None, 3), dtype=tf.uint8)])
    
    kn_feature_extractor_seq.add(kn_feature_extractor)
    
    return kn_feature_extractor_seq

def extract_features(mosaic, slide, network, batch_size):
    '''
    Function to extract features from the mosaic using DenseNet121
    Args:
        mosaic: list of patches
        slide: slide object
        batch_size: batch size for the model
    Returns:
        feature_queue: list of features
    '''
    # get the model
    if network == 'kimianet':
        model = get_kimianet_model(network_input_patch_width=1000)
    elif network == 'densenet121':
        model = get_dn121_model()
    else:
        raise ValueError('Network not supported')
    # get the objective power of the slide
    objective_power = int(slide.properties['openslide.objective-power'])
    # initialize the queues
    patch_queue = []
    feature_queue = []
    # iterate over the patches
    for patch in tqdm(mosaic):
        # this loc is at the objective power
        h, w = patch['wsi_loc']
        # Find the patch size at 20x
        patch_size_20x = int((objective_power/20.)*1000)
        # read the patch
        patch_region = slide.read_region((w, h), 0, (patch_size_20x, patch_size_20x)).convert('RGB')
        # Add to the queue
        patch_queue.append(np.array(patch_region))
        # if the queue is full, predict and add to the feature queue
        if len(patch_queue) == batch_size:
            feature_queue.extend(model.predict( np.array(patch_queue) ))
            patch_queue = []
    # if there are any patches left in the queue, predict and add to the feature queue
    if len(patch_queue) != 0:
        padded_arr = np.zeros((batch_size, patch_size_20x, patch_size_20x, 3), dtype=np.float32)
        padded_arr[:len(patch_queue), :, :, :] = np.array(patch_queue)
        feature_queue.extend(model.predict(padded_arr)[:len(patch_queue)])
    # return the feature queue
    return feature_queue    
        
class BoB:
    '''
    Class to represent the BoB of a slide
    '''
    def __init__(self, barcodes):
        '''
        Function to initialize the BoB
        Args:
            barcodes: list of bitarrays
        '''
        self.barcodes = [bitarray.bitarray(b.tolist()) for b in barcodes]
        
    def select_subset(self, n = 3):
        '''
        Function to select a subset of barcodes
        '''
        # shuffle the barcodes
        idx = np.arange(len(self.barcodes))
        np.random.shuffle(idx)
        idx = idx[:n]
        # return a new BoB object
        return BoB(barcodes=[self.barcodes[i] for i in idx])
    
    def distance(self, bob):
        '''
        Function to compute the distance between two BoBs
        '''
        # Initialize the total distance
        total_dist = []
        for feat in self.barcodes:
            # Compute the distance to all barcodes in the other BoB
            distances = [butil.count_xor(feat, b) for b in bob.barcodes]
            # Append the minimum distance
            total_dist.append(np.min(distances))
        # Return the median distance
        retval = np.median(total_dist)
        # Return the median distance
        return retval
    
def get_bob(slide_path, kmeans_clusters, tissue_threshold, percentage_selected, network, batch_size):
    '''
    Function to get the BoB from a slide
    Args:
        slide_path: path to the slide
        kmeans_clusters: number of clusters to use in kmeans
        tissue_threshold: threshold to use to select tissue patches
        percentage_selected: percentage of patches to select from each cluster
        batch_size: batch size to use for feature extraction
    Returns:
        bob: BoB object of the slide
    '''
    # Create the slide object
    slide = openslide.open_slide(slide_path)
    # Get the thumbnail
    thumbnail = slide.get_thumbnail((500, 500))
    # Get the tissue mask
    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = (cthumbnail.mean(axis=2) != 255)*1.
    # Get the patches
    patches = get_patches(slide, tissue_mask)
    flat_patches = get_flat_pathces(slide, patches, tissue_threshold)
    # Get the mosaic
    mosaic = get_mosaic(flat_patches, kmeans_clusters, tissue_threshold, percentage_selected)
    # Get the features
    features = extract_features(mosaic, slide, network, batch_size)
    # Get the BoB
    bob_raw = (np.diff(np.array(features), axis=1) < 0)*1
    bob = BoB(bob_raw)
    # Return the BoB
    return bob

def get_file_names_labels(csv_path):
    '''
    Function to get the file names and labels from the csv file
    Args:
        csv_path: path to the csv file
    Returns:
        file_paths: list of file paths
        labels: list of labels
    '''
    # Create a dataframe
    df = pd.read_csv(csv_path)
    # Get the file paths and labels
    file_paths = df['file_name'].values
    labels = df['label'].values
    # Return the file paths and labels
    return file_paths, labels


def predict_slide(
    new_bob: BoB,
    atlas_bobs: list,
    atlas_labels: list,
    top_n: int
    ):
    '''
    Function to predict the label of a new BoB
    Args:
        new_bob: BoB object of the new slide
        atlas_bobs: list of BoB objects of the atlas slides
        atlas_labels: list of labels of the atlas slides
        top_n: number of similar cases to use for prediction
    Returns:
        majority_vote: majority vote of the top_n similar cases
        top_n_labels: list of labels of the top_n similar cases
        top_n_distances: list of distances of the top_n similar cases
    '''
    # Error checking
    assert len(atlas_bobs) == len(atlas_labels), \
        'atlas_bobs and atlas_labels should have the same number of rows'
    
    # Compute the distances
    distances = np.array([new_bob.distance(bob) for bob in atlas_bobs])
    
    # Find the top_n similar cases   
    similar_indices = np.argsort(distances)
    
    sorted_labels = np.array(atlas_labels)[similar_indices]
    sorted_distances = distances[similar_indices]
    
    top_n_labels = sorted_labels[:top_n].tolist()
    top_n_distances = sorted_distances[:top_n].tolist()
       
    # Find the majority vote
    majority_vote = Counter(top_n_labels).most_common(1)[0][0]
    
    return majority_vote, sorted_labels.tolist(), sorted_distances.tolist()

def leave_one_out_test(
    atlas_bobs: list,
    atlas_labels: list,
    top_n: int
    ):
    '''
    Function to perform leave one out testing
    Args:
        atlas_bobs: list of BoB objects of the atlas slides
        atlas_labels: list of labels of the atlas slides
        top_n: number of similar cases to use for prediction
    Returns:
        y_true: list of true labels
        y_pred: list of predicted labels
    '''
    # Error checking
    assert len(atlas_bobs) == len(atlas_labels), 'atlas_bobs and atlas_labels should have the same number of rows'
    assert len(atlas_bobs) >= top_n, 'atlas_bobs should have equal or more rows than top_n'
    
    number_of_cases = len(atlas_bobs)
    
    # Initialize the variables
    original_atlas_bobs = atlas_bobs.copy()
    original_labels = atlas_labels.copy()
    y_true = []
    y_pred = []
    sorted_labels = []
    sorted_distances = []
    
    # Iterate through all the cases
    for out_index in tqdm(
        range(number_of_cases),
        total=number_of_cases,
        desc='Doing leave one out testing'
        ):
        # Get the test case and its label
        _test_case = original_atlas_bobs[out_index]
        _true_label = original_labels[out_index]
        
        # Remove the test case from the known cases
        _atlas_bobs = original_atlas_bobs[:out_index] + original_atlas_bobs[out_index + 1:]
        _atlas_labels = original_labels[:out_index] + original_labels[out_index + 1:]
        
        # Predict the label of the test case
        true = _true_label
        pred = predict_slide(_test_case, _atlas_bobs, _atlas_labels, top_n)
        
        # Append the true and predicted labels to the list
        y_true.append(true)
        y_pred.append(pred[0])
        sorted_labels.append(pred[1])
        sorted_distances.append(pred[2])
                           
    return y_true, y_pred, sorted_labels, sorted_distances




