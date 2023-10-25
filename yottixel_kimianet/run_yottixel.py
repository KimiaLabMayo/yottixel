import os
import logging
import warnings
import argparse
import time
import pickle
import numpy as np
# Suppressing warnings
warnings.filterwarnings("ignore")
# Parsing the arguments
parser = argparse.ArgumentParser(
    'Function to run the Yottixel pipeline for the WSI classification')
# Adding the arguments
parser.add_argument('--wsi_folder', type=str,
                    help='Folder containing the WSI files in svs format')
parser.add_argument('--csv_file', type=str,
                    help='CSV file containing the WSI file names and labels within the columns "file_name" and "label"')
parser.add_argument('--output_folder', type=str, default='./results/',
                    help='Folder where the output will be saved')
parser.add_argument('--kmeans_clusters', type=int, default=9,
                    help='Number of clusters for the kmeans')
parser.add_argument('--tissue_threshold', type=float,
                    default=0.7, help='Threshold for the tissue segmentation')
parser.add_argument('--percentage_selected', type=int,
                    default=5, help='Percentage of the selected patches')
parser.add_argument('--network', type=str, default='kimianet',
                    help='Network to use for the feature extraction')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Batch size for the patch extraction')
parser.add_argument('--top_n', nargs='+', default=['5'], help='Number of top patches to select')
parser.add_argument('--gpu', type=str, default='0', help='GPU device to use')
# Parsing the arguments
args = parser.parse_args()
# Getting the arguments
wsi_folder = args.wsi_folder
csv_file = args.csv_file
output_folder = args.output_folder
kmeans_clusters = args.kmeans_clusters
tissue_threshold = args.tissue_threshold
percentage_selected = args.percentage_selected
network = args.network
batch_size = args.batch_size
top_n = [int(_input) for _input in args.top_n]
gpu = args.gpu
# Validating the arguments
assert network in ['densenet121', 'kimianet'], \
    'The network argument must be either "densenet121" or "kimianet"'
assert os.path.isdir(wsi_folder), \
    'The wsi_folder argument must be a valid folder'
assert os.path.isfile(csv_file), \
    'The csv_file argument must be a valid file'
assert kmeans_clusters > 0, \
    'The kmeans_clusters argument must be greater than 0'
assert 0 < tissue_threshold < 1, \
    'The tissue_threshold argument must be between 0 and 1'
assert 0 < percentage_selected <= 100, \
    'The percentage_selected argument must be between 0 and 100'
assert batch_size > 0, \
    'The batch_size argument must be greater than 0'
assert np.all(np.array(top_n) > 0), \
    'The top_n argument must be greater than 0'
assert gpu in ['0', '1', '2', '3'], \
    'The gpu argument must be either "0", "1", "2" or "3" on Rhazes lab server'
# Setting the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# Setting the GPU memory growth and disabling the logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# Importing the libraries after setting the GPU
from helper_functions import *
from sklearn.metrics import confusion_matrix, classification_report
# Getting the slide paths
file_names, file_labels = get_file_names_labels(csv_file)
# Getting the slide paths
file_paths = [os.path.join(wsi_folder, file_name) for file_name in file_names]
# Creating the output folder
os.makedirs(output_folder, exist_ok=True)
# Creating the folder to save bob obejcts
os.makedirs(os.path.join(output_folder, 'bob_objects'), exist_ok=True)
# Initializing the error list
errors = []
# Create and configure logger
logging.basicConfig(filename=os.path.join(output_folder, 'log.txt'),
                    format=f'%(levelname)s %(asctime)s %(filename)s %(funcName)s %(lineno)d %(message)s',
                    filemode='w')
# Creating an object
logger = logging.getLogger()
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
# Test logger
logger.info("logger created")
########################### Starting the pipeline #############################
# Recording the start time
start_time = time.time()
logger.info(f'Starting the pipeline at {time.ctime()}')

# getting the bob object for all the slides
bobs = []
bob_lables = []
for s_index, slide_path in enumerate(file_paths):
    try:
        elapsed_time = time.time() - start_time
        print(f'Slide {s_index + 1} / {len(file_paths)} - {slide_path}')
        print(f'Total Elapsed Time: {elapsed_time/60:.0f} minutes')
        logger.info(
            f'Slide {s_index + 1} / {len(file_paths)} - {slide_path}'
            f'Total Elapsed Time: {elapsed_time/60:.0f} minutes'
            )
        # Getting the bob object for the current slide
        bob = get_bob(slide_path, kmeans_clusters, tissue_threshold,
                      percentage_selected, network, batch_size)
        # saving the bob object as a pickle file
        with open(os.path.join(output_folder, 'bob_objects', f'{file_names[s_index]}.pkl'), 'wb') as f:
            pickle.dump(bob, f)
        logger.info(f'Bob object for slide {slide_path} saved')
        # Appending the bob object to the list
        bobs.append(bob)
        bob_lables.append(file_labels[s_index])
    except Exception as e:
        print(f'Error in slide {slide_path}, {e}')
        errors.append((slide_path, e))
        logger.error(f'Error in slide {slide_path}, {e}')
        
# Recording the end time
indexing_end_time = time.time()

for _top_n in top_n:
    # Recording the lap time for indexing
    searching_lap_time = time.time()

    # Doing the leave one out testing
    logger.info(f'Starting the leave one out testing for {_top_n}')
    y_true, y_pred, sorted_labels, sorted_distances = leave_one_out_test(
        bobs, bob_lables, _top_n)
    logger.info(f'Leave one out testing done for {_top_n}')
    
    end_time = time.time()
    
    # saving the y_true, y_pred, sorted_labels and sorted_distances as dictionary pickle file in the output folder
    with open(os.path.join(output_folder, f'{_top_n}_test_results.pkl'), 'wb') as f:
        pickle.dump({
            'y_true': y_true,
            'y_pred': y_pred,
            'sorted_labels': sorted_labels,
            'sorted_distances': sorted_distances
        }, f)
    logger.info('y_true, y_pred, sorted_labels and sorted_distances saved')
    
    labels = np.unique(y_true)
    c_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report_text = classification_report(y_true, y_pred, output_dict=False)

    # creating the report containing the confusion matrix and the classification report
    # and all the parameters used for the pipeline
    report = (
        f'Parameters:\n'
        +f'wsi_folder: {wsi_folder}\n'
        +f'csv_file: {csv_file}\n'
        +f'output_folder: {output_folder}\n'
        +f'kmeans_clusters: {kmeans_clusters}\n'
        +f'tissue_threshold: {tissue_threshold}\n'
        +f'percentage_selected: {percentage_selected}\n'
        +f'batch_size: {batch_size}\n'
        +f'top_n: {_top_n}\n'
        +f'gpu: {gpu}\n'
        +f'Indexing Elapsed Time in minutes: {(indexing_end_time - start_time)/60:.0f} minutes\n'
        +f'Indexing Elapsed Time in seconds: {indexing_end_time - start_time:.2f} seconds\n'
        +f'Searching Elapsed Time in minutes for top {_top_n}: {(end_time - searching_lap_time)/60:.0f} minutes\n'
        +f'Searching Elapsed Time in seconds for top {_top_n}: {end_time - searching_lap_time:.2f} seconds\n'
        +f'Confusion Matrix:\n{c_matrix}\n'
        +f'Labels:\n{labels}\n'
        +f'Classification Report:\n{report_text}\n'
        +f'Errors:\n{errors}\n'
    )
    logger.info('Report created')

    # saving the report as a text file
    with open(os.path.join(output_folder, f'report_top_{_top_n}.txt'), 'w') as f:
        f.write(report)
    logger.info(f'Report saved for top {_top_n}')
    
# recording the end time
logger.info(f'Ending the pipeline at {time.ctime()}')
