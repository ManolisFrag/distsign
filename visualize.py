import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import random
from bokeh.io import output_notebook, output_file, push_notebook, show
from bokeh.plotting import figure, show
import re
import os
import os.path
from distsign import compare
from scipy.stats import kde

from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh import palettes
from bokeh.palettes import Spectral6, Inferno7, cividis, viridis, plasma
from bokeh.transform import linear_cmap
from bokeh.models.ranges import Range1d
from bokeh.layouts import column
from bokeh.palettes import brewer
from bokeh.palettes import mpl
from bokeh.palettes import d3
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt2d
from bokeh.models import BoxAnnotation, Toggle, Rect
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row 
from bokeh.plotting import figure
from bokeh.models import  ColumnDataSource,Range1d, LabelSet, Label
from bokeh.palettes import RdBu5,Set1, Spectral

output_notebook()

def create_dataframe(dir_to_search):
    list_json_files = []
    list_dirs = []
    list_img_files = []
    df = pd.DataFrame(columns=['Sequence', 'Json_file', 'Dominant_hand', 'Non_dominant_hand', 'Dominant_confidence', 'Non_dominant_confidence'])

    for dirpath, dirnames, filenames in os.walk(dir_to_search):
        for filename in [f for f in filenames if f.endswith(".json")]:
            list_json_files.append(os.path.join(dirpath, filename))
#             list_dirs.append(dirpath.replace('./data/', ''))
            list_dirs.append(dirpath)

    list_json_files.sort()
    list_dirs.sort()

    df['Sequence'] = list_dirs
    df['Json_file'] = list_json_files


    for i in range(df.shape[0]):
    #     print(str(df['Json_file'].iloc[i]))

        dominant, non_dominant, dominant_confidence, non_dominant_confidence =  compare.get_hands_from_json(str(df['Json_file'].iloc[i]))
#         fingers, conf = Handshape.handshape(str(df['Json_file'].iloc[i])).get_right_fingers_from_json
#         df['Fingers_coordinates'].iloc[i] = fingers
        df['Dominant_hand'].iloc[i] = dominant
        df['Non_dominant_hand'].iloc[i] = non_dominant
        df['Dominant_confidence'].iloc[i] = dominant_confidence
        df['Non_dominant_confidence'].iloc[i] = non_dominant_confidence
        
    df = df.drop(df.index[df["Dominant_hand"].isna()])
    return(df)


def create_dataframe_upper_body(dir_to_search, FLAG):
    from HSL import Handshape
    list_json_files_2 = []
    list_dirs_2 = []
    list_img_files_2 = []
    df_2 = pd.DataFrame(columns=['Sequence', 'Json_file','Pose'])

    exclude = set.union(set(['.ipynb_checkpoints']))
    for dirpath, dirnames, filenames in os.walk(dir_to_search, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in exclude]
    #     print(dirnames)
        for filename in [f for f in filenames if f.endswith(".json")]:
            list_json_files_2.append(os.path.join(dirpath, filename))
            list_dirs_2.append(dirpath.replace('./data/', ''))
            
    list_json_files_2.sort()
    list_dirs_2.sort()

    df_2['Sequence'] = list_dirs_2
    df_2['Json_file'] = list_json_files_2

    for i in range(df_2.shape[0]):
        if FLAG==1:
            pose_one =  Handshape.handshape(str(df_2['Json_file'].iloc[i])).get_right_arm_without_fingers
            df_2['Pose'].iloc[i] = pose_one
        elif FLAG == 0:
            pose_one =  Handshape.handshape(str(df_2['Json_file'].iloc[i])).get_upper_body_with_fingers
            df_2['Pose'].iloc[i] = pose_one
        elif FLAG == 2:
            pose_one =  Handshape.handshape(str(df_2['Json_file'].iloc[i])).get_right_wrist
            df_2['Pose'].iloc[i] = pose_one
    return(df_2)



def prepare_example_sign_normal(df, dirr, length_to_be_used, FLAG):
    df_1 = df[df.Sequence.str.contains(dirr)]["Pose"].values
    # 1 properly shape
    orig_shape = df_1.shape[0]
    array_sample = np.hstack(df_1)
    if FLAG== 0:
        # expected size (29,2)
        array_sample = array_sample.reshape((orig_shape,29,2))
        out = resize(array_sample, output_shape=[length_to_be_used,29,2])
    elif FLAG == 1:
        # expected size (5,2)
        array_sample = array_sample.reshape((orig_shape,5,2))
        out = resize(array_sample, output_shape=[length_to_be_used,5,2])
    elif FLAG == 2:
        # expected size (5,2)
        array_sample = array_sample.reshape((orig_shape,1,2))
        out = resize(array_sample, output_shape=[length_to_be_used,1,2])
    return out 


def plot_bok(X_3d, labels, string_of_path_to_be_removed, language_1, language_2):
    c1 = Set1[3][2] # green
    c2 = Set1[3][0] # red
    c3 = Set1[3][1] #blue

    labels = [x.replace(string_of_path_to_be_removed,'') for x in labels]

    colors_labels = np.empty_like(labels)
    for i in range(len(labels)):
        colors_labels[i] = c3
        if language_1 in labels[i]:
            colors_labels[i] = c1
        elif language_2 in labels[i]:
            colors_labels[i] = c2

    X_p, Y_p = zip(*X_3d)
    my_data_pca = {'x_values': X_p,
            'y_values': Y_p,
            'color_lab': colors_labels,
            'my_labels': labels}

    source_2 = ColumnDataSource(data=my_data_pca)
    TOOLTIPS = [
    ("(x,y)", "(@x_values, @y_values)"),]       

    p = figure(tooltips=TOOLTIPS)
    p.circle(x='x_values', y='y_values', source=source_2,color='color_lab')
    labels_p = LabelSet(x='x_values', y='y_values', text='my_labels', level='glyph',
                        text_font_size="6pt", 
                        x_offset=5, y_offset=5, source=source_2, render_mode='canvas')

    p.add_layout(labels_p)
    show(p)   



##### visuzalize with smoothing



def show_paths(dir_to_search):
    import matplotlib.image as mpimg 
    df = create_dataframe(dir_to_search)
    df = df.drop(df.index[df["Dominant_hand"].isna()])
    df.Sequence.unique()
    
    # non dominant hand
#     df_n = create_dataframe(dir_to_search)
    df_n = df.drop(df.index[df["Non_dominant_hand"].isna()])
    df_n.Sequence.unique()
   # Create data: 200 points
#     data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    data_n = df_n["Non_dominant_hand"]
    x_n, y_n = zip(*data_n)
    
    
    #####
    output_notebook()

    TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset, hover, save"
    palette=viridis

    # Get the number of colors we'll need for the plot.
    # colors = d3["Category40"][len(df.Sequence.unique())]
    colors = palette(len(df.Sequence.unique())+1)
    legend_it = []
    
    source = ColumnDataSource(data=dict(
        video=df['Sequence'].unique(),
    ))

    p = figure(tools=TOOLS , x_axis_label='x', y_axis_label='y', plot_width=1200)
    image = 'https://www.clipartkey.com/mpngs/m/32-329894_clip-art-png-outline-transparent-images-human-upper.png'
    p.image_url(url = [image], x=-1.1, y=2, w=2.4, h=2.5, anchor="bottom_left")
#     show(p)
    #adding the human stick figure layout
#     p.circle(0,0, size=80,fill_color="white", line_width=3)
#     p.rect(x=0, y=1.2, width=2, height=1.5, angle=0, fill_color="white",line_width=3 )

    
    color_counter = 0
    j=0
    for folder in df.Sequence.unique():
        per_video = df.loc[df['Sequence'] == folder]["Dominant_hand"]

        per_video_shape = per_video.shape[0]
        per_video = np.hstack(per_video.astype(object).values)
        per_video = per_video.reshape(per_video_shape,2)


        per_video = median_filter(per_video, size=3)

        per_video_confidence = df.loc[df['Sequence'] == folder]['Dominant_confidence']

        xs, ys = zip(*per_video)
        
        
        ############
        
        color = colors[color_counter]
        # add a line renderer with legend and line thickness
        c = p.line(xs, ys, line_color=color, muted_color=color, muted_alpha=0, line_width=2)
        c2 = p.line(x_n, y_n, line_color='blue', muted_color=color, muted_alpha=0, line_width=2)
    #     cp = p.circle(xs,ys,fill_color=color, line_color=color, size=6, alpha=per_video_confidence, muted_color=color, muted_alpha=0.05)

        color_counter += 1 
        legend_it.append((folder, [c]))  

#     p.x_range = Range1d(-1, 0)
#     p.y_range = Range1d(2, 0)

    p.x_range = Range1d(-2,2)
    p.y_range = Range1d(3.2,-1.5)



    legend = Legend(items=legend_it, location=(0, 0), click_policy='mute')
    legend.click_policy="mute"

    

    red_box = BoxAnnotation(bottom=2, top=1, fill_color='red', fill_alpha=0.04)
    green_box = BoxAnnotation(bottom=1, fill_color='green', fill_alpha=0.04)
    
    #background colors
#     p.add_layout(green_box)
#     p.add_layout(red_box)
    ####
    p.add_layout(legend,"left")
    p.legend.label_text_font_size = '8pt'

    show(p)
    
    
    
def show_heatmap(dir_to_search):
    import seaborn as sns
    sns.set()
    df = create_dataframe(dir_to_search)
    df = df.drop(df.index[df["Dominant_hand"].isna()])
    df.Sequence.unique()
#     output_notebook()

#     TOOLS = "crosshair,pan,wheel_zoom,box_zoom,reset, hover, save"
#     palette=viridis

#     # Get the number of colors we'll need for the plot.
#     # colors = d3["Category40"][len(df.Sequence.unique())]
#     colors = palette(len(df.Sequence.unique())+1)
#     legend_it = []
    
#     source = ColumnDataSource(data=dict(
#         video=df['Sequence'].unique(),
#     ))

#     p = figure(tools=TOOLS , x_axis_label='x', y_axis_label='y', plot_width=1200)
    
    
#     #adding the human stick figure layout
#     p.circle(0,0, size=80,fill_color="white", line_width=3)
#     p.rect(x=0, y=1.2, width=2, height=1.5, angle=0, fill_color="white",line_width=3 )

#     xs, ys = zip(*df["Dominant_hand"])
#     p.circle(xs,ys, size=5, color="navy", alpha=0.5)
#     p.x_range = Range1d(-2,2)
#     p.y_range = Range1d(3.2,-1.5)

#     show(p)


   # Create data: 200 points
#     data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    data = df["Dominant_hand"]
    x, y = zip(*data)
#     print(len(x))
    coords = pd.DataFrame(columns=['X', 'Y'])
    for i in range (len(x)):
        coords = coords.append({'X': x[i], 'Y': y[i]}, ignore_index=True)
    a = sns.jointplot(x='X',y='Y',data=coords, kind='kde',shade=True, ylim = (3.2,-1.5), xlim=(-2,2))

#     a = sns.kdeplot(coords.X, coords.Y, cmap="Reds", shade=True, bw=.15,ylim = (3.2,-1.5), xlim=(-2,2))

    a.ax_joint.plot([0],[0],'o',ms=60,mec='r',mfc='none')
#     a.ax_joint.plot([0],[0],'R',ms=100,mec='r',mfc='none')
#     # Add rectangle
    a.ax_joint.add_patch(
    patches.Rectangle(
    (-1, 0.6), # (x,y)
    2, # width
    1.5, # height
    # You can add rotation as well with 'angle'
    alpha=0.08, facecolor="red",linewidth=3, linestyle='solid'
    )
    )
    
    #left arm
    a.ax_joint.add_patch(
    patches.Rectangle(
    (-1, 0.6), # (x,y)
    0.3, # width
    2, # height
    angle=20,
    # You can add rotation as well with 'angle'
    alpha=0.08, facecolor="red",linewidth=3, linestyle='solid'
    )
    )
    
    #right arm
    a.ax_joint.add_patch(
    patches.Rectangle(
    (0.7, 0.7), # (x,y)
    0.3, # width
    2, # height
    angle=-20,
    # You can add rotation as well with 'angle'
    alpha=0.08, facecolor="red",linewidth=3, linestyle='solid'
    )
    )

    
#     plt.show()

def show_non_dominant_heatmap(dir_to_search):
    import seaborn as sns
    sns.set()
    df = create_dataframe(dir_to_search)
    df = df.drop(df.index[df["Non_dominant_hand"].isna()])
    df.Sequence.unique()
   # Create data: 200 points
#     data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    data = df["Non_dominant_hand"]
    x, y = zip(*data)
#     print(len(x))
    coords = pd.DataFrame(columns=['X', 'Y'])
    for i in range (len(x)):
        coords = coords.append({'X': x[i], 'Y': y[i]}, ignore_index=True)
        #cmap="rocket" works
    a = sns.jointplot(x='X',y='Y',data=coords, kind='kde',cmap="Reds",shade=False, ylim = (3.2,-1.5), xlim=(-2,2))

#     a = sns.kdeplot(coords.X, coords.Y, cmap="Reds", shade=True, bw=.15,ylim = (3.2,-1.5), xlim=(-2,2))

    a.ax_joint.plot([0],[0],'o',ms=60,mec='r',mfc='none')
#     a.ax_joint.plot([0],[0],'R',ms=100,mec='r',mfc='none')
#     # Add rectangle
    a.ax_joint.add_patch(
    patches.Rectangle(
    (-1, 0.6), # (x,y)
    2, # width
    1.5, # height
    # You can add rotation as well with 'angle'
    alpha=0.08, facecolor="red",linewidth=3, linestyle='solid'
    )
    )
    
    #left arm
    a.ax_joint.add_patch(
    patches.Rectangle(
    (-1, 0.6), # (x,y)
    0.3, # width
    2, # height
    angle=20,
    # You can add rotation as well with 'angle'
    alpha=0.08, facecolor="red",linewidth=3, linestyle='solid'
    )
    )
    
    #right arm
    a.ax_joint.add_patch(
    patches.Rectangle(
    (0.7, 0.7), # (x,y)
    0.3, # width
    2, # height
    angle=-20,
    # You can add rotation as well with 'angle'
    alpha=0.08, facecolor="red",linewidth=3, linestyle='solid'
    )
    )


def show_both_dominant_non_dominant_heatmap(dir_to_search):
    import seaborn as sns
    import matplotlib.image as mpimg 
    import warnings
    warnings.filterwarnings("ignore")
    sns.set()
    
    #non_dominant
    df = create_dataframe(dir_to_search)
    df = df.drop(df.index[df["Non_dominant_hand"].isna()])
    df.Sequence.unique()
    # Create data: 200 points
    #     data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    data = df["Non_dominant_hand"]
    x, y = zip(*data)
    #     print(len(x))
    coords = pd.DataFrame(columns=['X', 'Y'])
    for i in range (len(x)):
        coords = coords.append({'X': x[i], 'Y': y[i]}, ignore_index=True)
        #cmap="rocket" works


    # a = sns.jointplot(x='X',y='Y',data=coords, kind='kde',cmap="Reds",shade=False, ylim = (3.2,-1.5), xlim=(-2,2))
    # print(coords.head())
    #dominant
    df2 = create_dataframe(dir_to_search)
    df2 = df2.drop(df2.index[df2["Dominant_hand"].isna()])
    df2.Sequence.unique()
    # Create data: 200 points
    #     data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    data2 = df2["Dominant_hand"]
    x2, y2 = zip(*data2)
    #     print(len(x))
    coords2 = pd.DataFrame(columns=['X', 'Y'])
    for i in range (len(x2)):
        coords2 = coords2.append({'X': x2[i], 'Y': y2[i]}, ignore_index=True)
        #cmap="rocket" works

    #     g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)

    fig, ax = plt.subplots()
    sns.kdeplot(x=coords.X, y=coords.Y, cmap="Blues", ax=ax, shade=True, shade_lowest=False)
    sns.kdeplot(x=coords2.X, y=coords2.Y, cmap="Reds", ax=ax, shade=True, shade_lowest=False)
#     sns.plot([0],[0],'o',ms=60,mec='r',mfc='none')

    map_img = mpimg.imread('distsign/body4.png') 
 
    
    ax.set_xlim([-2,2])
    ax.set_ylim([3.2,-1.5])
    
    ax.imshow(map_img,
          aspect = ax.get_aspect(),
          extent = [-1.5,1.5] + [2.7,-0.6],
          zorder = 0,alpha=0.5) #put the map under the heatmap

    
def two_semantic_groups(group_1_path, group_2_path, string_of_path_to_be_removed,concept):
    import numpy as np
    group_1 = create_dataframe(group_1_path)
    group_2 = create_dataframe(group_2_path)
    dist = np.empty([len(group_1.Sequence.unique()), len(group_2.Sequence.unique())])
    for i in range(len(group_1.Sequence.unique())):
            for j in range(len(group_2.Sequence.unique())):            
                dist[i][j] = compare.one_sign_at_a_time(group_1.Sequence.unique()[i], group_2.Sequence.unique()[j])
    dist_df = pd.DataFrame(data=dist, index=group_1.Sequence.unique(), columns=group_2.Sequence.unique())
    fig = plt.figure(figsize=(10,8))
    dist_df = dist_df.reindex(sorted(dist_df.columns), axis=1)

    # dist_df[dist_df.columns] = dist_df.apply(lambda x: x.str.replace(string_of_path_to_be_removed,''))

    
    dist_df.columns = dist_df.columns.str.replace(string_of_path_to_be_removed,"")
    full_dict = dist_df.to_dict('index')

    new_index_values = [s[len('./'+ concept+'/'):] for s in dist_df.index.values]


    df_2 = pd.DataFrame(dist_df.columns[np.argsort(dist_df.values, axis=1)], 
                                   index=new_index_values)
    df_2[df_2.columns] = df_2.apply(lambda x: x.str.lstrip(string_of_path_to_be_removed))

    df_3 = np.sort(dist_df)[:, ::1]

    import seaborn as sns
    fig = plt.figure(figsize=(20,10))
    # df_3 = df_2.reindex(sorted(selected_languages), axis=1)
    # df_3 = df_2.reindex(sorted(selected_languages), axis=0)

    r = sns.heatmap(df_3, annot=df_2.values, cmap="Spectral_r", fmt='', annot_kws={"size": 8}, vmin=0, vmax=150)
    r.set_yticklabels(df_2.index.values, rotation=0)
    r.set_title("Similarities accross videos")
    # fig.savefig(concept+'.pdf')
    
    
    # uncomment for all signs
    # r = sns.heatmap(df_3[:,0:5], annot=df_2.iloc[:,0:5].values, cmap="Spectral_r", fmt='', annot_kws={"size": 8}, vmin=0, vmax=150)
    # r.set_yticklabels(df_2.iloc[:,0:5].index.values, rotation=0)
    # r.set_title("Similarities accross videos")
    # # fig.savefig(concept+'.pdf')
    # fig.savefig('all_signs.png')
    print("The mean dtw distance of one semantic field vs the other is : %s" % (dist_df.mean(axis = 1)).values.mean())
    return(df_2,df_3,full_dict)

def umap_visualization(group_1_path, group_2_path, language_1, language_2, string_of_path_to_be_removed, FLAG):
    '''
    Function to create a 2D visualization using the UMAP algorithm arguments

    :param group_1_path str: Path of the first language directory data
    :param group_2_path str: Path of the second language directory data
    :param language_1 str: Name of the language as used in the folder structure. This helps assign color in UMAP
    :param language_2 str: Name of the language as used in the folder structure. This helps assign color in UMAP
    :param string_of_path_to_be_removed str: In case of using absolute path the names of the signs will be large. You can remove parts of the strings by passing this parameter
    :param FLAG int: Select 0: to read all upper body and right hand fingers' joints, 1: to read right arm joints, 2: to read right hand wrist

    :return: Dataframe with signs and their UMAP calculated x,y coordinates
    '''

    import umap
    # print("Current Working Directory " , os.getcwd())

    # create dataframe with upper body and fingers joints per group
    group_1_df = create_dataframe_upper_body(group_1_path, FLAG)
    group_2_df = create_dataframe_upper_body(group_2_path, FLAG)

    #merge them
    df = group_1_df.append(group_2_df, sort=False).reset_index(drop=True)

    # reshape 
    labels = []
    data = []
    length_to_be_used = 86
    for seq in df.Sequence.unique():
        # change above number based on which data you want - check prepare_example_sign_normal
        pose_per_sequence = prepare_example_sign_normal(df, seq, length_to_be_used, FLAG)
        labels.append(seq)
        data.append(pose_per_sequence)
      
    if FLAG ==0:
        data = np.reshape(data,(len(data),length_to_be_used*29*2))
    elif FLAG == 1:
        data = np.reshape(data,(len(data),length_to_be_used*5*2))
    elif FLAG == 2:
        data = np.reshape(data,(len(data),length_to_be_used*2))

     #apply scaling
    data = StandardScaler().fit_transform(data)
    ######

    # for n in (2, 5):
    #     for d in (0.25, 0.5, 0.99):
    # print("Number of neighbors: {}, min_dist: {} ".format(n, d))
    metrix = ['correlation', 'cosine', 'wminkowski']
    for m in metrix:
        print("Metric is: ", m)
        reducer = umap.UMAP(n_neighbors=2, min_dist=0.35, metric=m, random_state=42, init='random')   
        
        # embedding = reducer.fit_transfrom(data)
        reducer.fit(data)
        embedding = reducer.transform(data)

        # plot_bok(embedding, labels)

        labels = np.array(labels)
        embedding = np.array(embedding)
        df_umap = pd.DataFrame({'Sequence': labels, '2d_point': list(embedding)}, columns=['Sequence', '2d_point'])

        plot_bok(df_umap["2d_point"], df_umap["Sequence"], string_of_path_to_be_removed,language_1, language_2)
    return(df_umap)
  


   
