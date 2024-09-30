import os
import sys
import argparse
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt



def plot_3d_trajectories(gt, est, ate, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(gt[0], gt[1], gt[2], color='blue', label='GT')
    ax.plot(est[0], est[1], est[2], color='red', label='Estimate')
    
    ax.legend()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'ATE: {ate:.3f}')
    plt.savefig(save_path)