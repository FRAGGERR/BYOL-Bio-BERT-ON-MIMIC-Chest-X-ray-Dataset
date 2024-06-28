import os
import pandas as pd
import cv2
import lungs_finder as lf
from tqdm import tqdm
from constants import *

def find_lung_region(image):
    default_bbox = [0, 0, 0, 0]
    right_lung_hog_rectangle = lf.find_right_lung_hog(image)
    left_lung_hog_rectangle = lf.find_left_lung_hog(image)
    
    right_lung_lbp_rectangle = lf.find_right_lung_lbp(image)
    left_lung_lbp_rectangle = lf.find_left_lung_lbp(image)
    
    right_lung_haar_rectangle = lf.find_right_lung_haar(image)
    left_lung_haar_rectangle = lf.find_left_lung_haar(image)
    
    bbox_r_hog = right_lung_hog_rectangle if right_lung_hog_rectangle is not None else default_bbox
    bbox_r_lbp = right_lung_lbp_rectangle if right_lung_lbp_rectangle is not None else default_bbox
    bbox_r_haar = right_lung_haar_rectangle if right_lung_haar_rectangle is not None else default_bbox
    
    bbox_l_hog = left_lung_hog_rectangle if left_lung_hog_rectangle is not None else default_bbox
    bbox_l_lbp = left_lung_lbp_rectangle if left_lung_lbp_rectangle is not None else default_bbox
    bbox_l_haar = left_lung_haar_rectangle if left_lung_haar_rectangle is not None else default_bbox
    
    x_r = min(bbox_r_hog[0], bbox_r_lbp[0], bbox_r_haar[0])
    y_r = min(bbox_r_hog[1], bbox_r_lbp[1], bbox_r_haar[1])
    width_r = min(bbox_r_hog[2], bbox_r_lbp[2], bbox_r_haar[2])
    height_r = min(bbox_r_hog[3], bbox_r_lbp[3], bbox_r_haar[3])
    
    x_l = min(bbox_l_hog[0], bbox_l_lbp[0], bbox_l_haar[0])
    y_l = min(bbox_l_hog[1], bbox_l_lbp[1], bbox_l_haar[1])
    width_l = min(bbox_l_hog[2], bbox_l_lbp[2], bbox_l_haar[2])
    height_l = min(bbox_l_hog[3], bbox_l_lbp[3], bbox_l_haar[3])
    
    bbox_r = [x_r, y_r, width_r, height_r]
    bbox_l = [x_l, y_l, width_l, height_l]
    
    return bbox_r, bbox_l

def extract_bboxes(df):
    right_lung_boxes = []
    left_lung_boxes = []
    for idx in tqdm(range(len(df))):
        img_idx = df[NIH_PATH_COL][idx]
        img_path = os.path.join(NIH_CXR_DATA_DIR , img_idx) 
        cv_image = cv2.imread(img_path)
        right_lung_box, left_lung_box = find_lung_region(cv_image)
        right_lung_boxes.append(right_lung_box)
        left_lung_boxes.append(left_lung_box)
    data = list(zip(right_lung_boxes, left_lung_boxes))
    return data

def pre_process_nih():
    xray_data = pd.read_csv(NIH_DATA_ENTRY_CSV)
    xray_data.drop(xray_data.columns.difference(['Image Index', 'Finding Labels']), axis=1, inplace=True)

    for label in NIH_TASKS:
        xray_data[label] = xray_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)


    # Initialize an empty dictionary to store image paths
    img_paths = {}

    # Traverse the directory structure
    for subdir in tqdm(NIH_CXR_DATA_DIR.iterdir()):
        if subdir.is_dir():
            images_dir = subdir / 'images'
            if images_dir.exists():
                for image_file in images_dir.glob('*.png'):
                    relative_path = os.path.join(subdir.name, 'images', image_file.name)
                    img_paths[image_file.name] = relative_path

    xray_data['Path'] = xray_data['Image Index'].map(img_paths)
    # Reorder columns to make 'Image Path' the second column
    cols = xray_data.columns.tolist()
    cols.insert(1, cols.pop(cols.index('Path')))
    xray_data = xray_data[cols]
    
    # Extract bounding boxes and add them to the dataframe
    bbox_data = extract_bboxes(xray_data)
    right_lung_boxes, left_lung_boxes = zip(*bbox_data)
    xray_data['Right_Lung_BBox'] = right_lung_boxes
    xray_data['Left_Lung_BBox'] = left_lung_boxes
    
    #Read original train and text lists from NIH    
    raw_train_df  = pd.read_csv(NIH_ORIGINAL_TRAIN_TXT, sep=" ", header=None)
    raw_test_df   = pd.read_csv(NIH_ORIGINAL_TEST_TXT, sep=" ", header=None)
    raw_train_df.columns = ['Image Index']
    raw_test_df.columns  = ['Image Index']
    
    # Create train_df and test_df by filtering xray_data with raw_train_df
    ttrain_df = xray_data[xray_data['Image Index'].isin(raw_train_df['Image Index'])]
    test_df = xray_data[xray_data['Image Index'].isin(raw_test_df['Image Index'])]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_df = train_df.drop(['Image Index'], axis=1)
    test_df = test_df.drop(['Image Index'], axis=1)
    train_df.to_csv(NIH_TRAIN_BBOX_CSV, index=False)
    test_df.to_csv(NIH_TEST_BBOX_CSV, index=False)
    
pre_process_nih()    