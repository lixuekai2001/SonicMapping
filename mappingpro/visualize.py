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
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
import plotly.io as pio


# Root directory of the project
ROOT_DIR = os.path.abspath("./Results")
#  abspath convert the relative path "../" or "."into absolute path.
# "../" is the upper level dir, so this code aim to get the upper path and convert it to abs path.
# "." is the current dir
sys.path.append(ROOT_DIR)  

from coalpro import utils

############################################################
#  Image processing
############################################################

def mapping_2D(matrix, mag_times, data_name, sample_name, colorbar_unit,direction, colorbar_range):
    # Plot the 2D mapping 
#     plt.figure(figsize=(6, 6))
    x_new,y_new,z_new = utils.matrix_interpo_2D(matrix,mag_times,method ='nearest')
    norm = Normalize(vmin=colorbar_range[0], vmax=colorbar_range[1]) if colorbar_range else None
    ax = plt.gca()
    im = ax.imshow(z_new, cmap='plasma_r', interpolation='nearest', norm=norm)
    avg = round(z_new.mean(),2)

    if mag_times >3:
        anno = 0
    else: anno = 1

    #Add text annotations for each value
    if anno:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='k')

    plt.title( data_name+" on "+direction+" face, "+sample_name+' ('+str(avg)+colorbar_unit+')')
    if direction == 'X':
        plt.xlabel("Y")
        plt.ylabel("Z")
    elif direction == 'Y':
        plt.xlabel("X")
        plt.ylabel("Z")
    else:
        plt.xlabel("X")
        plt.ylabel("Y")
    plt.xticks([])
    plt.yticks([])
    # Create an axis for the colorbar that matches the size of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(colorbar_unit)
    cbar.ax.xaxis.set_label_position('bottom')
    return avg

def plot_mapping(df,mag_times,data_name,sample_name,colorbar_unit):
    # anno=0/1, if annotation is on, mag_times: >3, 
    # Create a 3x1 subplot for the XYZ directions
    result_folder = os.path.join(ROOT_DIR, "2D/Result_2D_"+sample_name)

    # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"'{result_folder}' Result folder has been created.")
    # else:
    #     print(f"'{result_folder}' Figures will be saved in Result folder.")
        
    fig, axs = plt.subplots(1, 3, figsize=(18, 8),dpi=300)
    direc = ['X','Y','Z']
    Avg = []

    for i in np.arange(len(df)):
        plt.sca(axs[i])
        a = mapping_2D(df[i],  mag_times, data_name ,sample_name ,colorbar_unit, direction = direc[i], colorbar_range=[df.min(), df.max()])
        # mapping_2D(matrix, mag_times = 10, data_name = "Young's Modulus",direction = 'X', sample_name = 'Coal-39', colorbar_range=None,colorbar_unit='GPa')
        # If you want to provide a default value for an argument, make sure it comes after all the required arguments.
        Avg.append(a)
    plt.savefig(result_folder+'/'+sample_name+"_"+data_name+"_matrix_"+str(mag_times)+".png")
    return Avg


def scatter3D(sample_name,data_name,all_coords,all_yme,unit):
    result_folder = os.path.join(ROOT_DIR, "Result_3D_"+sample_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Result folder has been created:'{result_folder}' ")
    else:
        print(f"Figures will be saved in Result folder:'{result_folder}' ")
    #
    color_min = min(all_yme)
    color_max = max(all_yme)
    fig = go.Figure(data=[go.Scatter3d(
        x=all_coords[:, 0],
        y=all_coords[:, 1],
        z=all_coords[:, 2],
        mode='markers',
        marker=dict(
            size=13,
            color=all_yme,  # Set color to the value of the matrix
            colorscale='plasma_r',  # Use the 'plasma_r' colorscale
            symbol = 'circle',
            colorbar=dict(title=unit),
            opacity=0.8,
            cmin=color_min,
            cmax=color_max
            ))
    ])
    fig.update_layout(
        width=800,
        height=600,
        scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    html_content = fig.to_html(full_html=False)
    with open(result_folder+'/'+sample_name+"_"+data_name+"_scatter3D.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    # fig.show()

def c_2D_plot(df, Lx,Ly,Lz, method, data_name, sample_name, unit,plot_type,selected_face):
    # Plot the 2D mapping on 3 x 3 x 3 matrix
    # method =['kriging','RBF','linear','cubic','nearest']
    # direction = ['x','y','z']
    grid_x, grid_y, grid_2d = utils.matrix_interpo_xyz(df, Lx, Ly, Lz, selected_face ,method)
    colorbar_range=[df.min(), df.max()]
    norm = Normalize(vmin=colorbar_range[0], vmax=colorbar_range[1]) if colorbar_range else None

    if selected_face in ['X', 'x']:
        xlabel="Y (mm)"
        ylabel="Z (mm)"
        x_e = Ly
        y_e = Lz
    elif selected_face in ['Y', 'y']:
        xlabel="X (mm)"
        ylabel="Z (mm)"
        x_e = Lx
        y_e = Lz
    else:
        xlabel="X (mm)"
        ylabel="Y (mm)"
        x_e = Lx
        y_e = Ly
    extent = [0, x_e, y_e, 0]
    ax = plt.gca()
    if plot_type in ['contour']:
        im = ax.imshow(grid_2d, aspect='equal',cmap='plasma_r', interpolation='nearest', norm=norm,extent=extent)
        cs = plt.contour(grid_x, grid_y, grid_2d, levels=5, colors='white')
        ax.clabel(cs, inline=True, fontsize=12)  # Add contour labels
        # ax.invert_yaxis()
    else:
        im = ax.imshow(grid_2d, aspect='equal',cmap='plasma_r', interpolation='nearest', norm=norm,extent=extent)
        # ax.invert_yaxis()
    # 反转 y 轴
    
    avg = round(grid_2d.mean(),2)

    plt.title( sample_name+','+data_name+' (avg='+str(avg)+unit+')'+" on "+selected_face+" face, "+method)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Create an axis for the colorbar that matches the size of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(unit)
    cbar.ax.xaxis.set_label_position('bottom')

def c_mapping(df, Lx,Ly,Lz,data_name,sample_name,colorbar_unit,method,plot_type='mapping'):
    # data_name = 'E', sample_name='c1',colorbar_unit = 'GPa', method ='rbf'
    # Create a 3x1 subplot for the XYZ directions
    result_folder = os.path.join(ROOT_DIR, "2D/Result_2D_"+sample_name)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"'{result_folder}' Result folder has been created.")
    # else:
    #     print(f"Figures will be saved in Result folder: '{result_folder}' ")
        
    fig, axs = plt.subplots(1, 3, figsize=(18, 8),dpi=300)
    direc = ['X','Y','Z']
    plt.figure(figsize=(20, 6))
    for i in np.arange(len(df)):
        plt.sca(axs[i])
        c_2D_plot(df, Lx,Ly,Lz, method,data_name, sample_name, colorbar_unit,plot_type,selected_face=direc[i])
        # If you want to provide a default value for an argument, make sure it comes after all the required arguments.
    plt.tight_layout()
    # plt.gca().set_aspect('equal')
    plt.savefig(result_folder+'/'+sample_name+"_"+data_name+"_"+str(method)+'_'+str(plot_type)+".png")
    # plt.show()

def c_2D_plot_norm(df, Lx,Ly,Lz, method, data_name, sample_name, unit,plot_type,selected_face):
    # Plot the 2D mapping on 3 x 3 x 3 matrix
    # method =['kriging','RBF','linear','cubic','nearest']
    # direction = ['x','y','z']
    grid_x, grid_y, grid_2d = utils.matrix_interpo_xyz(df, Lx, Ly, Lz, selected_face ,method)
    colorbar_range=[0, 1]
    norm = Normalize(vmin=colorbar_range[0], vmax=colorbar_range[1]) if colorbar_range else None

    if selected_face in ['X', 'x']:
        xlabel="Y (mm)"
        ylabel="Z (mm)"
        x_e = Ly
        y_e = Lz
    elif selected_face in ['Y', 'y']:
        xlabel="X (mm)"
        ylabel="Z (mm)"
        x_e = Lx
        y_e = Lz
    else:
        xlabel="X (mm)"
        ylabel="Y (mm)"
        x_e = Lx
        y_e = Ly
    extent = [0, x_e, y_e, 0]
    ax = plt.gca()
    if plot_type in ['contour']:
        im = ax.imshow(grid_2d/grid_2d.max(), aspect='equal',cmap='plasma_r', interpolation='nearest', norm=norm,extent=extent)
        cs = plt.contour(grid_x, grid_y, grid_2d/grid_2d.max(), levels=5, colors='white')
        ax.clabel(cs, inline=True, fontsize=12)  # Add contour labels
        # ax.invert_yaxis()
    else:
        im = ax.imshow(grid_2d/grid_2d.max(), aspect='equal',cmap='plasma_r', interpolation='nearest', norm=norm,extent=extent)
        # ax.invert_yaxis()
    # 反转 y 轴
    
    avg = round(grid_2d.mean(),2)

    plt.title( sample_name+','+data_name+' (avg='+str(avg)+unit+')'+" on "+selected_face+" face, "+method)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return avg

    # Create an axis for the colorbar that matches the size of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Normalized '+data_name)
    cbar.ax.xaxis.set_label_position('bottom')

def c_mapping_norm(df, Lx,Ly,Lz,data_name,sample_name,colorbar_unit,method,plot_type='mapping'):
    # data_name = 'E', sample_name='c1',colorbar_unit = 'GPa', method ='rbf'
    # Create a 3x1 subplot for the XYZ directions
    result_folder = os.path.join(ROOT_DIR, "2D/Result_2D_"+sample_name)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"'{result_folder}' Result folder has been created.")
    else:
        print(f"Figures will be saved in Result folder: '{result_folder}' ")
        
    fig, axs = plt.subplots(1, 3, figsize=(18, 8),dpi=300)
    direc = ['X','Y','Z']
    plt.figure(figsize=(20, 6))
    average = []
    for i in np.arange(len(df)):
        plt.sca(axs[i])
        avg = c_2D_plot_norm(df, Lx,Ly,Lz, method,data_name, sample_name, colorbar_unit,plot_type,selected_face=direc[i])
        average.append(avg)
        # If you want to provide a default value for an argument, make sure it comes after all the required arguments.
    plt.tight_layout()
    # plt.gca().set_aspect('equal')
    plt.savefig(result_folder+'/'+sample_name+"_"+data_name+"_"+str(method)+'_'+str(plot_type)+"_norm.png")
    # plt.show()

def contour3D(matrix,data_name,sample_name,colorbar_unit,df_avg,result_folder):
    x, y, z = np.indices(matrix.shape)
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=matrix.flatten(),
        isomin=matrix.min(),
        isomax=matrix.max(),
        colorscale='plasma_r',  # Use the 'plasma_r' colorscale
        colorbar=dict(
            title=colorbar_unit,
            x=1,  # Set the x position of the color bar
            y=0.5,  # Set the y position of the color bar
            len=0.75,  # Set the length of the color bar
            thickness=20,  # Set the thickness of the color bar
            orientation='v'  # Set the orientation of the color bar
        ),

        opacity=0.5, # Adjust the opacity for better visualization
        surface_count=17, # Adjust for more detailed surfaces
    ))
    fig.update_layout(
        title= sample_name+", "+data_name+"_avg = "+str(round(df_avg,2))+colorbar_unit,
        width=800,
        height=800,
        scene=dict(
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        zaxis_title='Z (mm)'
    ))
    camera = dict(eye=dict(x=1.2,y=-2,z=0.8))
    fig.update_layout(scene_camera=camera)
    pio.write_image(fig,result_folder+"/"+sample_name+"_"+data_name+'_contour.png',scale = 3)
    # fig.show()

    html_content = fig.to_html(full_html=False)
    with open(result_folder+"/"+sample_name+"_"+data_name+'_contour.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Plot saved to *.html")
    
    ## Plotly exports images at a default resolution of 96 DPI. To achieve a different DPI, you can use the formula: scale=desired DPI/default DPI
    # For a desired DPI of 300:scale=300/96≈3.125
   
def volume3D(matrix,data_name,sample_name,colorbar_unit,df_avg,result_folder):
    x, y, z = np.indices(matrix.shape)
    fig = go.Figure(data=go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=matrix.flatten(),
        isomin=matrix.min(),
        isomax=matrix.max(),
        colorscale='plasma_r',
        colorbar=dict(
            title=colorbar_unit,
            x=1,  # Set the x position of the color bar
            y=0.5,  # Set the y position of the color bar
            len=0.75,  # Set the length of the color bar
            thickness=20,  # Set the thickness of the color bar
            orientation='v'  # Set the orientation of the color bar
        ),
    ))
    fig.update_layout(
        title= sample_name+" "+data_name+"_avg = "+str(round(df_avg,2))+colorbar_unit,
        width=800,
        height=800,
        scene=dict(
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        zaxis_title='Z (mm)')
    )
    camera = dict(eye=dict(x=1.2,y=-2,z=0.8))
    fig.update_layout(scene_camera=camera)
    pio.write_image(fig,result_folder+"/"+sample_name+"_"+data_name+'_3D.png',scale = 3)
    # fig.show()
    html_content = fig.to_html(full_html=False)
    with open(result_folder+"/"+sample_name+"_"+data_name+'_3D.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Plot saved to *.html")

def plot_3D(df,mag_times,data_name,sample_name,colorbar_unit):
    result_folder = os.path.join(ROOT_DIR, "3D/Result_3D_"+sample_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Result folder has been created:'{result_folder}' ")
    else:
        print(f"Figures will be saved in Result folder:'{result_folder}' ")
    #
    df3,df_avg = utils.volume_data(df)
    interpolated_matrix = utils.matrix_interpo_3D(df3, mag_times)
    volume3D(interpolated_matrix,data_name,sample_name,colorbar_unit,df_avg,result_folder)
    contour3D(interpolated_matrix,data_name,sample_name,colorbar_unit,df_avg,result_folder)
    return df_avg

def plot_3D_xyz(df,Lx,Ly,Lz,data_name,sample_name,unit,method = 'rbf'):
    result_folder = os.path.join(ROOT_DIR, "3D/Result_3D_"+sample_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"Result folder has been created:'{result_folder}' ")
    else:
        print(f"Figures will be saved in Result folder:'{result_folder}' ")
    #
    df_interpo,df_volume, df_avg = utils.volume_data_xyz(df,Lx,Ly,Lz,method)
    volume3D(df_volume,data_name,sample_name,unit,df_avg,result_folder)
    contour3D(df_volume,data_name,sample_name,unit,df_avg,result_folder)
    return df_avg

def unisize_2Dmapping(df,sample_name):
    result_folder = os.path.join(ROOT_DIR, "2D/Result_2D_"+sample_name)
    unit = ['m/s','m/s','GPa','_','GPa','GPa']
    for mag_time in [3]:
        j = -1
        x = []
        y = []
        z = []
        for i in list(df):
            j=j+1
            print(j)
            print("plot:"+list(df)[j]+"_"+str(mag_time))
            Avg = plot_mapping(df[i],mag_time,list(df)[j],sample_name,unit[j])
            x.append(Avg[0])
            y.append(Avg[1])
            z.append(Avg[2])
        Avg_sum = {
        "Items": list(df),
        "x_avg":x,
        "y_avg":y,
        "z_avg":z
        }
        Avg_df = pd.DataFrame(Avg_sum)
    Avg_df.to_csv(result_folder+'/'+sample_name+"_avg_"+str(mag_time)+".csv", index=False)
    return Avg_df

def Avg_xyz(df,mag_time):
    Avg = []
    for i in np.arange(len(df)):
        x_new,y_new,z_new = utils.matrix_interpo_2D(df[i],mag_time,method ='nearest')
        avg = round(z_new.mean(),2)
        Avg.append(avg)
    return Avg

def anisotropy(df,sample_name):
    result_folder = os.path.join(ROOT_DIR, "2D/Result_2D_"+sample_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    # unit = ['m/s','m/s','GPa','_','GPa','GPa']
    for mag_time in [3]:
        j = -1
        x = []
        y = []
        z = []
        for i in list(df):
            j=j+1
            Avg = Avg_xyz(df[i],mag_time)
            x.append(Avg[0])
            y.append(Avg[1])
            z.append(Avg[2])
        Avg_sum = {
        "Items": list(df),
        "x_avg":x,
        "y_avg":y,
        "z_avg":z
        }
        Avg_df = pd.DataFrame(Avg_sum)
    Avg_df.to_csv(result_folder+'/'+sample_name+"_avg_mag"+str(mag_time)+".csv", index=False)
    return Avg_df
            
def fullsize_2Dmapping(df,Lx,Ly,Lz,sample_name):
    unit = ['m/s','m/s','GPa','_','GPa','GPa']
    for method in ['rbf']:
        j = -1
        for i in list(df):
            j=j+1
            print(j)
            print("plot:"+list(df)[j]+"_"+method)
            c_mapping(df[i], Lx,Ly,Lz,list(df)[j],sample_name,unit[j],method,'contour')
            c_mapping(df[i], Lx,Ly,Lz,list(df)[j],sample_name,unit[j],method)

def fullsize_2Dmapping_norm(df,Lx,Ly,Lz,sample_name):
    unit = ['m/s','m/s','GPa','_','GPa','GPa']
    for method in ['rbf']:
        j = -1
        for i in list(df):
            j=j+1
            print(j)
            print("plot:"+list(df)[j]+"_"+method)
            c_mapping_norm(df[i], Lx,Ly,Lz,list(df)[j],sample_name,unit[j],method,'contour')
            c_mapping_norm(df[i], Lx,Ly,Lz,list(df)[j],sample_name,unit[j],method)

def fullsize_3Dmapping(df,Lx,Ly,Lz,sample_name):
    unit = ['m/s','m/s','GPa','_','GPa','GPa']
    for method in ['rbf']:
        j = -1
        for i in list(df):
            j=j+1
            print(j)
            print("plot:"+list(df)[j]+"_"+method)
            plot_3D_xyz(df[i],Lx,Ly,Lz,list(df)[j],sample_name,unit[j])

def plot_2d_data(data_dir,a=1,b=1):
    #
    files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    sample_names = [file.split('_')[0] for file in files if file.endswith('.csv')]
    #
    Avg_xyz = []
    for i in np.arange(len(files)):
        print('Start processing:'+str(sample_names[i]))
        df,Lx,Ly,Lz,density  = utils.df_clean_new(data_dir+files[i],'linear',a,b)
        Avg_df = unisize_2Dmapping(df,sample_names[i])
        fullsize_2Dmapping(df,Lx,Ly,Lz,sample_names[i])
        Avg_xyz.append(Avg_df)
    return Avg_xyz

def plot_3d_data(data_dir,a=1,b=1):
    #
    files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    sample_names = [file.split('_')[0] for file in files if file.endswith('.csv')]
    #
    for i in np.arange(len(files)):
        print('Start processing:'+str(sample_names[i]))
        df,Lx,Ly,Lz,density  = utils.df_clean_new(data_dir+files[i],'linear',a,b)
        fullsize_3Dmapping(df,Lx,Ly,Lz,sample_names[i])

def plot_vpvs(combined_df,title):
    result_folder = os.path.join(ROOT_DIR, "Result_fitting")
        # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"'{result_folder}' Result folder has been created.")
    else:
        print(f"'{result_folder}' Figures will be saved in Result folder.")

    Vp = [float(num) for num in combined_df.Vp]
    Vs = [float(num) for num in combined_df.Vs]

    x,y,linear_y_fit,slope_linregress,intercept_linregress,r_value,xx,exponential_y_fit,a,b,r2 = utils.fitting_vpvs(Vp,Vs)
    #
    plt.figure(figsize=(7, 6))  # Width: 10 inches, Height: 5 inches
    plt.scatter(x, y, label='Data', c='purple', alpha = 0.7 ,edgecolor='black',s=30 )
    if intercept_linregress>0:
        plt.plot(x, linear_y_fit, color='k', label=f'Linear fit: y={slope_linregress:.2f}x+{intercept_linregress:.2f} (R²={r_value**2:.2f})')
    else:
        plt.plot(x, linear_y_fit, color='k', label=f'Linear fit: y={slope_linregress:.2f}x{intercept_linregress:.2f} (R²={r_value**2:.2f})')

    plt.plot(xx, exponential_y_fit, color='red', label=f'Expon fit: y={a:.2f}exp({b:.5f}x) (R²={r2:.2f})')
    plt.xlabel('Vp (m/s)',fontsize=12)
    plt.ylabel('Vs (m/s)',fontsize=12)
    # Rotate x-axis labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    # plt.xlim(20, 120)  # Set the x-axis to range from 2 to 8
    # plt.ylim(20, 120)  # Set the x-axis to range from 2 to 8
    plt.title(title)
    plt.legend(loc = 'upper left')
    plt.savefig(result_folder+'/'+title+'.png')
    return slope_linregress,intercept_linregress
