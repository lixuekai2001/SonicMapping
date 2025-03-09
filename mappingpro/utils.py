"""
3D Ultrasonic mapping data for cubic material
calculation functions.

Copyright (c) 2024
Author: Jimmy Li
Email: jimmy.li@uq.edu.au
Licensed under the MIT License (see LICENSE for details)

"""

import os
import sys
import random
import itertools
import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from numpy.polynomial.polynomial import Polynomial

from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata, RBFInterpolator
from scipy.interpolate import RegularGridInterpolator

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
#  abspath convert the relative path "../" into absolute path.
# "../" is the upper level dir, so this code aim to get the upper path and convert it to abs path.

sys.path.append(ROOT_DIR)  

############################################################
#  data processing
############################################################

def vs_fitting(vp,vs,fitting_method,a,b):
    # Concatenate the two DataFrames horizontally
    vpvs = pd.concat([vp, vs], axis=1)
    # Assign new column names
    vpvs.columns = ['P', 'S']
    if fitting_method == 'linear':
        Vs = vpvs.apply(lambda row: b + a * row['P'] if pd.isna(row['S']) else row['S'], axis=1)
    else:
        Vs = vpvs.apply(lambda row: a * np.exp(b * row['P']) if pd.isna(row['S']) else row['S'], axis=1) 
    return Vs

def df_clean_new(file_path,fitting_method,a,b):
    # new_csv_file_path = 'coal-39.csv'
    data_new = pd.read_csv(file_path)
    ## Get w,x,y,z
    w = data_new.iloc[11,1]
    w = pd.to_numeric(w, errors='coerce')
    Lx = data_new.iloc[14,1]
    Lx = pd.to_numeric(Lx, errors='coerce')
    Ly = data_new.iloc[15,1]
    Ly = pd.to_numeric(Ly, errors='coerce')
    Lz = data_new.iloc[16,1]
    Lz = pd.to_numeric(Lz, errors='coerce')
    # sample density (replace with actual density)
    density = w/Lx/Ly/Lz*1e6

    data_new.columns = data_new.iloc[0]
    #
    xps = data_new.iloc[1:10,1:3]
    yps = data_new.iloc[1:10,3:5]
    zps = data_new.iloc[1:10,5:7]
    #
    xps = xps.applymap(pd.to_numeric, errors='coerce')
    yps = yps.applymap(pd.to_numeric, errors='coerce')
    zps = zps.applymap(pd.to_numeric, errors='coerce')
    # use the fitting equation to calculate the STT_s 

    df_clean = pd.concat([xps, yps, zps], axis=1)
    df_clean.columns = ['x_p', 'x_s', 'y_p', 'y_s', 'z_p', 'z_s']

    # Calculate velocities
    df_clean['v_x_p'] = Lx / df_clean['x_p']*1000
    df_clean['v_x_s'] = Lx / df_clean['x_s']*1000
    df_clean['v_y_p'] = Ly / df_clean['y_p']*1000
    df_clean['v_y_s'] = Ly / df_clean['y_s']*1000
    df_clean['v_z_p'] = Lz / df_clean['z_p']*1000
    df_clean['v_z_s'] = Lz / df_clean['z_s']*1000

    df_clean['v_x_s'] = vs_fitting(df_clean['v_x_p'],df_clean['v_x_s'],fitting_method,a,b)
    df_clean['v_y_s'] = vs_fitting(df_clean['v_y_p'],df_clean['v_y_s'],fitting_method,a,b)
    df_clean['v_z_s'] = vs_fitting(df_clean['v_z_p'],df_clean['v_z_s'],fitting_method,a,b)

    # Calculate Young's Modulus (GPa) and Poisson's Ratio
    df_clean['E_x'] = density * (3 * df_clean['v_x_p']**2 - 4 * df_clean['v_x_s']**2)
    df_clean['E_y'] = density * (3 * df_clean['v_y_p']**2 - 4 * df_clean['v_y_s']**2)
    df_clean['E_z'] = density * (3 * df_clean['v_z_p']**2 - 4 * df_clean['v_z_s']**2)

    df_clean['nu_x'] = (df_clean['v_x_p']**2 - 2 * df_clean['v_x_s']**2) / (2 * (df_clean['v_x_p']**2 - df_clean['v_x_s']**2))
    df_clean['nu_y'] = (df_clean['v_y_p']**2 - 2 * df_clean['v_y_s']**2) / (2 * (df_clean['v_y_p']**2 - df_clean['v_y_s']**2))
    df_clean['nu_z'] = (df_clean['v_z_p']**2 - 2 * df_clean['v_z_s']**2) / (2 * (df_clean['v_z_p']**2 - df_clean['v_z_s']**2))

    # Calculate Shear Modulus (G)
    df_clean['G_x'] = density * df_clean['v_x_s']**2
    df_clean['G_y'] = density * df_clean['v_y_s']**2
    df_clean['G_z'] = density * df_clean['v_z_s']**2

    # Calculate Bulk Modulus (K)
    df_clean['K_x'] = density * (df_clean['v_x_p']**2 - (4/3) * df_clean['v_x_s']**2)
    df_clean['K_y'] = density * (df_clean['v_y_p']**2 - (4/3) * df_clean['v_y_s']**2)
    df_clean['K_z'] = density * (df_clean['v_z_p']**2 - (4/3) * df_clean['v_z_s']**2)

    # Create a new DataFrame for the mechanical properties
    # STTp_df = df_clean[['x_p', 'y_p', 'z_p']].copy()
    # STTs_df = df_clean[['x_s', 'y_s', 'z_s']].copy()
    Vp_df = df_clean[['v_x_p', 'v_y_p', 'v_z_p']].copy()
    Vs_df = df_clean[['v_x_s', 'v_y_s', 'v_z_s']].copy()
    E_df = df_clean[['E_x', 'E_y', 'E_z']].copy()
    nu_df = df_clean[['nu_x', 'nu_y', 'nu_z']].copy()
    G_df = df_clean[['G_x', 'G_y', 'G_z']].copy()
    K_df = df_clean[['K_x', 'K_y', 'K_z']].copy()

    # STTp = convert_matrix(STTp_df)
    # STTs = convert_matrix(STTs_df)
    Vp = convert_matrix(Vp_df)
    Vs = convert_matrix(Vs_df)
    E = convert_matrix(E_df)*1e-9
    nu = convert_matrix(nu_df)
    G = convert_matrix(G_df)*1e-9
    K = convert_matrix(K_df)*1e-9

    df = {
    # 'STTp': STTp,
    # 'STTs': STTs,
    'Vp': Vp,
    'Vs': Vs,
    'E': E,
    'nu': nu,
    'G': G,
    'K': K
    }

    return df,Lx,Ly,Lz,density

def convert_matrix_old(E_df):
    # this old function to work with old 3D configureation where,
    # X+ face from left (X-) to right (X+) and from top (Z+) to bottom(Z-) have 123456789 
    # Z+ face from lett (X-) to right (X+) and from top (Y+) to bottom(Y-) have 123456789
    # Y- face from lett (X-) to right (X+) and from top (Y+) to bottom(Y-) have 321564987

    # Reshape each column into a 3x3 matrix
    E_x_matrix = E_df.iloc[:,0].values.reshape(3, 3)
    E_y_matrix = E_df.iloc[:,1].values.reshape(3, 3)
    E_y_matrix = np.fliplr(E_y_matrix) # this is due to the measurement on y faces has a mirror flip with other faces
    E_z_matrix = E_df.iloc[:,2].values.reshape(3, 3)

    # Combine the three 3x3 matrices into a single 3x3x3 matrix
    E_combined_matrix = np.array([E_x_matrix, E_y_matrix, E_z_matrix])

    return E_combined_matrix

def convert_matrix(E_df):
    # this new function to work with new 3D configureation where,
    # X+ face from left (X-) to right (X+) and from top (Z+) to bottom(Z-) have 123456789 
    # Z+ face from lett (X-) to right (X+) and from top (Y+) to bottom(Y-) have 123456789
    # Y- face from lett (X-) to right (X+) and from top (Y+) to bottom(Y-) have 123456789 

    # Reshape each column into a 3x3 matrix
    E_x_matrix = E_df.iloc[:,0].values.reshape(3, 3)
    E_y_matrix = E_df.iloc[:,1].values.reshape(3, 3)
    E_z_matrix = E_df.iloc[:,2].values.reshape(3, 3)

    # Combine the three 3x3 matrices into a single 3x3x3 matrix
    E_combined_matrix = np.array([E_x_matrix, E_y_matrix, E_z_matrix])

    return E_combined_matrix

def generate_coords_3D(E, Lx, Ly, Lz, face):
    # E is 3x3x3 matrix, E[0],E[1],E[2] are X,Y,Z face values
    coords = []
    yme = []
    for i in range(3):
        for j in range(3):
            if face == 'z-':#bottom
                coords.append(((2 * (j + 1) - 1) * Lx / 6, (2 * (i + 1) - 1) * Ly / 6, 0))
                yme.append(E[2][i, j])  # 
            elif face == 'z+':#top
                coords.append(((2 * (j + 1) - 1) * Lx / 6, (2 * (i + 1) - 1) * Ly / 6, Lz))
                yme.append(E[2][i, j])  #
            elif face == 'y-':  # front face first row from top has same z values, so i is z
                coords.append(((2 * (j + 1) - 1) * Lx / 6, 0, (2 * (i + 1) - 1) * Lz / 6))
                yme.append(E[1][i, j])  # 
            elif face == 'y+': # back
                coords.append(((2 * (j + 1) - 1) * Lx / 6, Ly, (2 * (i + 1) - 1) * Lz / 6))
                yme.append(E[1][i, j])  # 
            elif face == 'x-': #left
                coords.append((0, (2 * (j + 1) - 1) * Ly / 6, (2 * (i + 1) - 1) * Lz / 6))
                yme.append(E[0][i, j])  # 
            elif face == 'x+':# right
                coords.append((Lx, (2 * (j + 1) - 1) * Ly / 6, (2 * (i + 1) - 1) * Lz / 6))
                yme.append(E[0][i, j])  # 
    return np.array(coords), np.array(yme)

def get_3D_values(E, Lx, Ly, Lz):
    # faces = ['bottom', 'top', 'front', 'back', 'left', 'right']
    faces = ['z-', 'z+', 'y-', 'y+', 'x-', 'x+']
    all_coords = []
    all_yme = []

    # 生成所有面的测量点
    for face in faces:
        coords, yme = generate_coords_3D(E, Lx, Ly, Lz, face)
        all_coords.append(coords)
        all_yme.append(yme)

    # 合并所有面的测量点
    all_coords = np.vstack(all_coords)
    all_yme = np.hstack(all_yme)
    return all_coords,all_yme

def generate_coords_2D(E, Lx, Ly, Lz, selected_face):
    # E is 3x3x3 matrix, E[0],E[1],E[2] are X,Y,Z face values
    coords = []
    yme = []
    for i in range(3):
        for j in range(3):
            if selected_face in ['Z', 'z']:#bottom
                coords.append(((2 * (j + 1) - 1) * Lx / 6, (2 * (i + 1) - 1) * Ly / 6))# x, y
                yme.append(E[2][i, j])  # 
            elif selected_face in ['Y', 'y']:  # front face first row from top has same z values, so i is z
                coords.append(((2 * (j + 1) - 1) * Lx / 6,  (2 * (i + 1) - 1) * Lz / 6))# x, z
                yme.append(E[1][i, j])  # 
            elif selected_face in ['X', 'x']: #left
                coords.append(((2 * (j + 1) - 1) * Ly / 6, (2 * (i + 1) - 1) * Lz / 6))# y, z
                yme.append(E[0][i, j])  # 

    coords = np.array(coords)
    yme = np.array(yme)
    # new_order = [6, 7, 8, 3, 4, 5, 0, 1, 2]
    # xyz face start from upper left, so it need up-down flip
    # coords = coords[new_order]
    return coords, yme

def matrix_interpo_xyz(df, Lx, Ly, Lz, selected_face, method='kriging'):
    # df is 3 x 3 x 3 matrix, such as E or STTp
    # method =['kriging','rbf','linear','cubic','nearest']
    # selected_face = ['x','y','z']
    coords, yme = generate_coords_2D(df, Lx, Ly, Lz, selected_face)
    if selected_face in ['Z', 'z']:
        grid_x = np.linspace(0, Lx, int(Lx))
        grid_y = np.linspace(0, Ly, int(Ly))
    elif selected_face in ['Y', 'y']:
        grid_x = np.linspace(0, Lx, int(Lx))
        grid_y = np.linspace(0, Lz, int(Lz))
    else:  # 'left', 'right' # X axis
        grid_x = np.linspace(0, Ly, int(Ly))
        grid_y = np.linspace(0, Lz, int(Lz))

    coords_x = coords[:,0]
    coords_y = coords[:,1]
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    if method in ["kriging","Kriging"]:
        kriging_2d = OrdinaryKriging(coords_x, coords_y, yme, variogram_model='spherical')
        grid_2d, ss_2d = kriging_2d.execute('grid', grid_x, grid_y)
    elif method in ['rbf' or "RBF"] :
        rbf_2d = RBFInterpolator(coords[:,:2], yme)
        grid_2d = rbf_2d(np.column_stack([grid_xx.ravel(), grid_yy.ravel()])).reshape(grid_xx.shape)
    else:
        # method =['linear','cubic','nearest']
        x = ([min(grid_x),max(grid_x)/2,max(grid_x)])
        y = ([max(grid_y),max(grid_y)/2,min(grid_y)])# this is because the sequence of measurement is from up to down and from left to right.
        xx, yy = np.meshgrid(x, y)
        grid_2d = griddata((xx.ravel(), yy.ravel()), yme, (grid_xx, grid_yy), method=method)
        grid_2d = np.flipud(grid_2d)
    return grid_x, grid_y, grid_2d

def volume_data(E):
    # Calculate the volume properties
    volumes = np.zeros((3, 3, 3))
    E_z = np.rot90(E[2], k=-1) # k= -1 顺时针旋转90度。
    for i in range(3):
        for j in range(3):
            for k in range(3):
                volumes[i, j, k] = (E[0][j, k] + E[1][j, i] + E_z[i, k])/3

    # Calculate the average volume
    average_volume = np.mean(volumes)
    average_volume = round(average_volume,2)
    print("Average Volume:", average_volume)
    return volumes,average_volume

def volume_data_xyz(E_2D,Lx,Ly,Lz,method):
    E = []
    for selected_face in ['X','Y','Z']:
        grid_x, grid_y, grid_2d = matrix_interpo_xyz(E_2D, Lx, Ly, Lz, selected_face ,method)
        E.append(grid_2d)
        
    n_x = E[2].shape[1]
    n_y = E[2].shape[0]
    n_z = E[0].shape[0]
    volumes = np.zeros((n_x, n_y, n_z))
    
    E_z = np.rot90(E[2], k=-1) # k= -1 顺时针旋转90度。
    for i in range(n_x):
        for j in range(n_z):
            for k in range(n_y):
                volumes[i, k, j] = (E[0][j, k] + E[1][j, i] + E_z[i, k])/3
    avg = round(volumes.mean(),2)
    print("Average Volume:", avg)
    E1 = np.flip(volumes)
    E2 = np.fliplr(E1)
    E3 = np.flipud(E2)
    return E,E3, avg

def matrix_interpo_3D(matrix, N):
    # Define the original grid
    original_grid = [np.linspace(0, 1, num=3) for _ in range(3)]
    # Create interpolating function
    interpolating_function = RegularGridInterpolator(original_grid, matrix)
    # Define the new grid
    new_grid = [np.linspace(0, 1, num=N) for _ in range(3)]
    new_grid_points = np.array(np.meshgrid(*new_grid)).T.reshape(-1, 3)
    # Interpolate data
    interpolated_matrix = interpolating_function(new_grid_points).reshape(N, N, N)
    return interpolated_matrix

def matrix_interpo_2D(matrix,N,method ='cubic'):
    # Define the original grid
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    x, y = np.meshgrid(x, y)
    # Define the new grid
    x_new = np.linspace(0, matrix.shape[0] - 1, N)
    y_new = np.linspace(0, matrix.shape[1] - 1, N)
    x_new, y_new = np.meshgrid(x_new, y_new)
    # Interpolate the data to the new grid
    z_new = griddata((x.ravel(), y.ravel()), matrix.ravel(), (x_new, y_new), method='cubic')
        # Ensure z_new has no NaN values by using nearest interpolation to fill missing values
    mask = np.isnan(z_new)
    z_new[mask] = griddata((x.ravel(), y.ravel()), matrix.ravel(), (x_new[mask], y_new[mask]), method='nearest')
    if N>3:
        x_new = x_new
        y_new = y_new
        z_new = z_new
    else:
        x_new = x
        y_new = y
        z_new = matrix
    return x_new,y_new,z_new

def read_vpvs(file_path):    
    data_new = pd.read_csv(file_path,header=None)
    Lx = data_new.iloc[15,1]
    Lx = pd.to_numeric(Lx, errors='coerce')
    Ly = data_new.iloc[16,1]
    Ly = pd.to_numeric(Ly, errors='coerce')
    Lz = data_new.iloc[17,1]
    Lz = pd.to_numeric(Lz, errors='coerce')

    xps = data_new.iloc[2:11,1:3]
    yps = data_new.iloc[2:11,3:5]
    zps = data_new.iloc[2:11,5:7]
    #
    xps = xps.applymap(pd.to_numeric, errors='coerce')
    yps = yps.applymap(pd.to_numeric, errors='coerce')
    zps = zps.applymap(pd.to_numeric, errors='coerce')
    
    df_clean = pd.concat([xps, yps, zps], axis=1)
    df_clean.columns = ['x_p', 'x_s', 'y_p', 'y_s', 'z_p', 'z_s']
    v = pd.DataFrame([])
    # Calculate velocities
    v['v_x_p'] = Lx / df_clean['x_p']*1000
    v['v_x_s'] = Lx / df_clean['x_s']*1000
    v['v_y_p'] = Ly / df_clean['y_p']*1000
    v['v_y_s'] = Ly / df_clean['y_s']*1000
    v['v_z_p'] = Lz / df_clean['z_p']*1000
    v['v_z_s'] = Lz / df_clean['z_s']*1000

    df = v.dropna()
    return df

def combine_vpvs(directory,title):
    result_folder = os.path.join(ROOT_DIR, "Results/Result_fitting")
        # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"'{result_folder}' Result folder has been created.")
    else:
        print(f"'{result_folder}' Figures will be saved in Result folder.")

    # List all files in the directory
    # files = os.listdir(directory)
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    # sample_names = [file.split('_')[0] for file in files if file.endswith('.csv')]
    # Initialize a list to hold DataFrames
    dataframes = []
    all_sonic = []
    # Loop through the files and process CSV files
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        df = read_vpvs(file_path)
        sonic = df.copy()
        sonic.insert(0,'sample_id',file_name.split('_')[0])
        dataframes.append(df)
        all_sonic.append(sonic)
    #     print(f"Read {file_name} successfully.")
    # Optional: Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True) # the elements are string format
    combined_sonic = pd.concat(all_sonic, ignore_index=True) # the elements are string format
    # the combined_sonic is calculated to find the wrong velocity data and locate its poision.
    # df = combined_sonic
    # numeric_df = df.copy()
    # numeric_df.iloc[:, 1:] = numeric_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # # Define the value threshold
    # value_threshold = 10000

    # # Find rows containing values greater than the threshold
    # rows = numeric_df[numeric_df.iloc[:, 1:] > value_threshold].dropna(how='all')

    # # Merge the results back with the original DataFrame to keep the first column
    # result = df.loc[rows.index]

    # print(f"Rows containing values greater than {value_threshold}:\n", result)

    # Flatten the DataFrame into a single array
    flattened = combined_df.to_numpy().flatten()
    # Reshap the flattened array into the desired format with 2 columns
    # Number of rows is len(flattened) divided by 2
    num_rows = len(flattened) // 2
    reshaped_array = flattened[:num_rows * 2].reshape(num_rows, 2)
    # Create a new DataFrame from the reshaped array
    combined_df2 = pd.DataFrame(reshaped_array, columns=['Vp','Vs'])
    combined_df2.to_csv(result_folder+"/"+title+"_combined_VP_VS.csv")
    # Vp = [float(num) for num in combined_df2.Vp]
    # Vs = [float(num) for num in combined_df2.Vs]
    return combined_df2

def fitting_vpvs(Vp,Vs):
    from scipy.stats import linregress
    x = np.array(Vp)
    y = np.array(Vs)
    # Linear fitting using scipy.stats.linregress
    slope_linregress, intercept_linregress, r_value, p_value, std_err = linregress(x, y)
    linear_y_fit = slope_linregress*x+intercept_linregress
    # Expon fitting
    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    a = np.exp(intercept)
    b = slope
    y_fit = a * np.exp(b * x)
    # Calculate R^2
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    # for plot
    xx = np.arange(min(x),max(x),10)
    exponential_y_fit = a * np.exp(b * xx)
    return x,y,linear_y_fit,slope_linregress,intercept_linregress,r_value,xx,exponential_y_fit,a,b,r2

def df_avg(df,sample_name):
    avg = []
    for item in list(df):
        avg.append(round(df[item].mean(),2))
    # avg = pd.DataFrame([avg])
    # avg.index = [sample_name]
    return avg

def dict_shape(my_dict):
    # Find the number of key-value pairs
    num_keys = len(my_dict)

    # Find the length of the lists (assuming all lists are the same length)
    if num_keys > 0:
        num_elements = len(next(iter(my_dict.values())))
    else:
        num_elements = 0
    # Print the result
    print("The dictionary has", num_keys, "keys and each key has", num_elements, "elements in its list.")

import pickle
def data_extract(data_dir,title,a=1,b=1):
    result_folder = os.path.join(ROOT_DIR, 'Results/Result_data/'+title+'/')
            # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"'{result_folder}' Result folder has been created.")
    else:
        print(f"'{result_folder}' Figures will be saved in Result folder.")
    #
    files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    sample_names = [file.split('_')[0] for file in files if file.endswith('.csv')]
    data_avg = []
    for i in np.arange(len(files)):
        print("Processing: "+str(files[i]))
        df,Lx,Ly,Lz,density  = df_clean_new(data_dir+files[i],'linear',a,b)
        #collect avergage data
        avg = df_avg(df,sample_names[i])
        data_avg.append(avg+[Lx,Ly,Lz,density])
        with open(result_folder+sample_names[i]+'.pkl', 'wb') as pickle_file:
            pickle.dump(df, pickle_file)
    data_avg = pd.DataFrame(data_avg)
    data_avg.index = sample_names
    data_avg.columns = list(df)+['Lx','Ly','Lz','Density']
    data_avg.to_csv(result_folder+title+'_sonic_avg.csv')