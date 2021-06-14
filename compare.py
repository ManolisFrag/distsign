import pandas as pd
import re
import os
import os.path
import json
import numpy as np
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter, median_filter
from dtw import *


def get_hands_from_json(json_file_path):
    """
    function to open a json file and return the x,y positions of the hands.
    Provide the path to the json file
    """
    with open(json_file_path) as f:
        loaded_json = json.load(f)
#         for dest in loaded_json["people"][0]:
#             print(dest)
        if len(loaded_json["people"]) == 0:
            print("EMPTY")
            print(json_file_path)
            return("NaN","NaN","NaN","NaN")
            
        raw_coords = loaded_json["people"][0]["pose_keypoints_2d"]
        dominant_array_confidence = raw_coords[14]
        non_dominant_array_confidence = raw_coords[23]
        
        raw_coords = np.delete(raw_coords, np.arange(2, len(raw_coords), 3))
        raw_coords = np.reshape(raw_coords, (25,2))
        
  
        raw_coords = np.array([raw_coords[0], raw_coords[1], raw_coords[2], raw_coords[3], raw_coords[4], raw_coords[5], raw_coords[6], raw_coords[7], raw_coords[15], raw_coords[16], raw_coords[17], raw_coords[18]])
        
#         print("coords are: %s" % raw_coords[4])
        test_list = np.array([0,0])
#         print(test_list)
        if np.array_equal(raw_coords[4], test_list):
#             print("Zero hands %s" % raw_coords[4])
            return(float('NaN'),float('NaN'),float('NaN'),float('NaN'))
        else:
            hands = raw_coords

            hands2 = hands - hands[0][0]
            #get the x values
            handsx = hands2[:,0]
            #get y values
            handsy = hands2[:,1]
            scaledY = handsy - hands2[0][1]
            scaledHands = np.array((handsx,scaledY))
            scaledHands = scaledHands.T

            dist = distance.euclidean(scaledHands[2], scaledHands[5])
            final_scaled = scaledHands/dist



            dominant_array = np.array(final_scaled[4]) #change to raw_coords[4] for COCO
            #print(dominant_array)
            non_dominant_array = np.array(final_scaled[7]) # change to raw_coords[7] for COCO

            if (np.all((hands[7] == 0))):
    #             print("Left hand not found")
                non_dominant_array = float('NaN')

            return(dominant_array, non_dominant_array, dominant_array_confidence, non_dominant_array_confidence)


def one_sign_at_a_time(*args, **kwds):
    """
    Function to compare two signs based on the directories that their openpose data are stored

    Parameters
    ----------
    sign1_dir : absolute directory of first sign
       
    sign2_dir : absolute directory of second sign
    """

    sign1_dir_l = args[0]
    sign2_dir_l = args[1]

#     os.chdir('../')
    # print('directory of function: ',os.getcwd())
    # print(os.listdir(os.getcwd()))

    # for folder, subfolders, files in os.walk(sign1_dir_l):
    #     print('files, ',files)
    
    #sign 1
    list_json_files_1 = []
    list_dirs_1 = []
    list_img_files_1 = []
    df1 = pd.DataFrame(columns=['Sequence', 'Json_file', 'Dominant_hand', 'Non_dominant_hand', 'Dominant_confidence', 'Non_dominant_confidence'])

    #sign 2
    list_json_files_2 = []
    list_dirs_2 = []
    list_img_files_2 = []
    df2 = pd.DataFrame(columns=['Sequence', 'Json_file', 'Dominant_hand', 'Non_dominant_hand', 'Dominant_confidence', 'Non_dominant_confidence'])

    #sign 1
    for dirpath, dirnames, filenames in os.walk(sign1_dir_l):
        # print('filenames are', filenames)
        for filename in [f for f in filenames if f.endswith(".json")]:
            list_json_files_1.append(os.path.join(dirpath, filename))
            list_dirs_1.append(dirpath)

    list_json_files_1.sort()
    list_dirs_1.sort()

    df1['Sequence'] = list_dirs_1
    df1['Json_file'] = list_json_files_1

    #sign 2
    for dirpath, dirnames, filenames in os.walk(sign2_dir_l):
        # print('filenames are', filenames)
        for filename in [f for f in filenames if f.endswith(".json")]:
            list_json_files_2.append(os.path.join(dirpath, filename))
            list_dirs_2.append(dirpath)

    list_json_files_2.sort()
    list_dirs_2.sort()

    df2['Sequence'] = list_dirs_2
    df2['Json_file'] = list_json_files_2

    #sign 1 dataframe
    for i in range(df1.shape[0]):
        dominant, non_dominant, dominant_confidence, non_dominant_confidence =  get_hands_from_json(str(df1['Json_file'].iloc[i]))
        df1['Dominant_hand'].iloc[i] = dominant
        df1['Non_dominant_hand'].iloc[i] = non_dominant
        df1['Dominant_confidence'].iloc[i] = dominant_confidence
        df1['Non_dominant_confidence'].iloc[i] = non_dominant_confidence

    #sign 2 dataframe
    for i in range(df2.shape[0]):
        dominant, non_dominant, dominant_confidence, non_dominant_confidence =  get_hands_from_json(str(df2['Json_file'].iloc[i]))
        df2['Dominant_hand'].iloc[i] = dominant
        df2['Non_dominant_hand'].iloc[i] = non_dominant
        df2['Dominant_confidence'].iloc[i] = dominant_confidence
        df2['Non_dominant_confidence'].iloc[i] = non_dominant_confidence

    df1 = df1.drop(df1.index[df1["Dominant_confidence"] < 0.2])
    df2 = df2.drop(df2.index[df2["Dominant_confidence"] < 0.2])

    sign_1_video_path = df1["Dominant_hand"].dropna().reset_index(drop=True)
    sign_2_video_path = df2["Dominant_hand"].dropna().reset_index(drop=True)
    

    ex1 = np.array(sign_1_video_path.to_list())
    ex1 = ex1.reshape((int(len(ex1)),2))   
    ex1 = median_filter(ex1,size=3, mode="wrap")
    

    ex2 = np.array(sign_2_video_path.to_list())
    ex2 = ex2.reshape((int(len(ex2)),2))   
    ex2 = median_filter(ex2,size=3, mode="wrap")

    # removing preparation
    non_preparation_array_ex1 = []
    non_preparation_array_ex2 = []

    # print('ex1 shape is', ex1.shape[0])
    for k in range(ex1.shape[0]):
        if ex1[k,1] > 1:
            non_preparation_array_ex1.append(k)    
    ex1 = np.delete(ex1, (non_preparation_array_ex1), axis=0)
    
    for l in range(ex2.shape[0]):
        if ex2[l,1] > 1:
            non_preparation_array_ex2.append(l)    
    ex2 = np.delete(ex2, (non_preparation_array_ex2), axis=0)
    
    # I changed to from x=ex2 , y=ex1. Perhaps put it back again?
    final_distance = dtw(x=ex1, y=ex2,open_begin=True,open_end=True,step_pattern="asymmetric").distance

    return(final_distance)
            

            







