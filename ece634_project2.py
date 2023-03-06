import os
import cv2
import copy
import math
import glob
import numpy as np
from matplotlib import pyplot as plt


#Part 1: Generalized Lloyd algorithm

#This function splits an image into multiple blocks. It's the same as used in project 1.
def split_img(img_pths, block_size):
    all_blocks = []
    for img_pth in img_pths:
        img = cv2.imread(img_pth, 0).astype(np.float32)
        assert img.shape[0] % block_size[0] == 0  #the image needs to be perfectly divisible into blocks
        assert img.shape[1] % block_size[1] == 0
        n_rows = img.shape[0] // block_size[0]
        n_cols = img.shape[1] // block_size[1]
        split_rows = np.vsplit(img, img.shape[0]//block_size[0])
        for rows in split_rows:
            split_columns = np.hsplit(rows, img.shape[1]//block_size[1]) #this, along with the vsplit function above extracts grids from the array
            for cols in split_columns:
                all_blocks.append(np.ravel(cols)) #individual blocks are stored as elements of a list 
    all_blocks = np.array(all_blocks)
    
    return all_blocks, n_rows, n_cols

#This function takes all the blocks in the training set and initializes the codebook by
#selecting randomly from among the blocks. The codebook length can be passed as a parameter.
def initialize_codebook(codebook_length, all_blocks):        
    random_indices = np.random.randint(low = 0, high = all_blocks.shape[0], size = (codebook_length,))
    random_codebook = all_blocks[random_indices]
    return random_codebook

#This function computes the minimum distortion by comparing the blocks with all elements in the codebook
#likewise, it stores the indices of the minimum distortion so that the blocks can be clustered and the 
#codebook can be updated in the next step
#last but not least, it also outputs the average distortion for checking with the stopping criterion
def calculate_min_indices(all_blocks, codebook):
    all_min_distortion = []
    all_min_idx = []
    for block_nbr in range(all_blocks.shape[0]):
        min_distortion = 999999999
        min_idx = None
        for code_idx in range(codebook.shape[0]):
            distortion = np.sum((all_blocks[block_nbr] - codebook[code_idx])**2)
            if distortion < min_distortion:
                min_distortion = distortion
                min_idx = code_idx
        assert min_idx is not None
        all_min_distortion.append(min_distortion)
        all_min_idx.append(min_idx)
    all_min_distortion = np.array(all_min_distortion)
    avg_distortion = np.mean(all_min_distortion)
    all_min_idx = np.array(all_min_idx)
    
    return all_min_distortion, avg_distortion, all_min_idx
        

#this function updates the codebook until the stopping criterion is reached. it calls on the above
#functions.
def update_codebook(all_blocks, codebook, prev_avg_distortion, stopping_threshold):
    all_min_distortion, avg_distortion, all_min_idx = calculate_min_indices(all_blocks, codebook)
    updated_codebook = copy.deepcopy(codebook) #so that the original codebook is preserved
    rel_diff = np.abs(avg_distortion - prev_avg_distortion)/prev_avg_distortion
    if (rel_diff < stopping_threshold):  #stopping criterion
        stop_training = True #this is passed on to another function from which training is stopped
    else: #only if training stopping criterion is not reached
        for code_idx in range(codebook.shape[0]):
            relevant_codes = all_blocks[all_min_idx == code_idx] #cluster per the minimum indices
            if relevant_codes.shape[0] != 0: #skip if no block falls within a code's cluster
                avg_codes = np.mean(relevant_codes, axis = 0) #centroid/average of the cluster
                updated_codebook[code_idx] = avg_codes #update codebook as the average
        stop_training = False #return this if training stopping criterion is not reached
            
    return updated_codebook, avg_distortion, stop_training

#this function performs training on a set of images and returns the final codebook. 
#other than training image paths, codebook length, block size, and stopping threshold
#are also passed as parameters and can be changed according to use-case.
def train(img_pths, codebook_length, block_size, stopping_threshold):
    prev_avg_distortion = 999999 #placeholder intial average distortion before the training begins
    all_blocks, _, _ = split_img(img_pths, block_size) #split the image into blocks
    codebook = initialize_codebook(codebook_length, all_blocks) #initialize the codebook
    while True: #update the codebook for several iterations
        codebook, prev_avg_distortion, stop_training = update_codebook(all_blocks, codebook, prev_avg_distortion, stopping_threshold)
        if stop_training: #exit if stopping criterion is reached
            break
            
    return codebook


#this function takes the test image path and the final codebook, and returns the images for visualization,
#as well as all the original blocks
def test(img_pth, codebook):
    vector_dimension = codebook.shape[1]
    block_size = (int(np.sqrt(vector_dimension)), int(np.sqrt(vector_dimension))) #as a tuple for row/column numbers
    all_blocks, n_rows, n_cols = split_img([img_pth], block_size) #split the test image
    old_img = cv2.imread(img_pth, 0).astype(np.float32) #preserve original image
    new_img = np.empty((old_img.shape[0], old_img.shape[1])).astype(np.float32) 
    
    all_min_distortion, avg_distortion, all_min_idx = calculate_min_indices(all_blocks, codebook) #calculate minimum indices
        
    for block_nbr in range(all_blocks.shape[0]):    
        row_nbr = block_nbr // n_cols
        col_nbr = block_nbr - n_cols * row_nbr
        new_img[row_nbr * block_size[0]: (row_nbr + 1) * block_size[0], col_nbr * block_size[1]: (col_nbr + 1) * block_size[1]] = codebook[all_min_idx[block_nbr]].reshape((block_size[0], block_size[1])) #replace with codebook vector that results in minimum distortion and reshape into 2d shape for reconstructing the image
    old_img = old_img.astype(np.uint8) #8 bit rgb image
    new_img = new_img.astype(np.uint8) #8 bit rgb image
    
    return all_blocks, old_img, new_img


#this is the main function that performs all the tasks in the project
if __name__ == '__main__':
    
    root_dir = '/kaggle/input/ece634-project2/sample_image'
    train_img_pths = [f'{root_dir}/airplane.tif', f'{root_dir}/baboon.png', f'{root_dir}/barbara.png', f'{root_dir}/boat.png',
                  f'{root_dir}/zelda.png', f'{root_dir}/text.tif', f'{root_dir}/peppers.tif', f'{root_dir}/sails.png', 
                  f'{root_dir}/monarch.png', f'{root_dir}/girl.png']
    
    block_size = (4, 4)
    stopping_threshold = 0.05
    
    for self_training in [True, False]:
        for test_img_name in ['pentagon.tif', 'goldhill.png', 'cameraman.tif']:
            test_img_pth = f'{root_dir}/{test_img_name}'
            if self_training:
                train_img_pths = [test_img_pth]
            for codebook_length in [128, 256]:
                save_pth = f'./self_training_{self_training}_test_image_name_{test_img_name}_codebook_length_{codebook_length}.jpg'
                codebook = train(img_pths = train_img_pths, codebook_length = codebook_length, block_size = block_size, stopping_threshold = stopping_threshold)
                all_blocks, old_img, new_img = test(test_img_pth, codebook)
                psnr = 20 * math.log10(255) - 10 * math.log10(np.sum((new_img - old_img) ** 2)/(new_img.shape[0] * new_img.shape[1]))
                print(f'For self training = {self_training}, test image name = {test_img_name}, codebook length = {codebook_length}, the PSNR is {psnr}')
                plt.imshow(new_img, cmap = 'gray')
                plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) #Courtesy https://www.geeksforgeeks.org/how-to-remove-ticks-from-matplotlib-plots/
                plt.savefig(f'{save_pth}')
                plt.close()






#Part 2: DCT Transform
#Theory learned from the lecture videos and https://www.math.cuhk.edu.hk/~lmlui/dct.pdf

#This function splits the image into blocks after scaling it so that the range is from -128 to 127.
def scale_and_split(img_pth, dimension):
    all_blocks = []
    img = cv2.imread(img_pth, 0).astype(np.float32)
    assert img.shape[0] % dimension == 0  #the image needs to be perfectly divisible into blocks
    assert img.shape[1] % dimension == 0
    n_rows = img.shape[0] // dimension
    n_cols = img.shape[1] // dimension
    
    #According to https://www.math.cuhk.edu.hk/~lmlui/dct.pdf, image should be scaled
    scaled_img = img - 128
    split_rows = np.vsplit(scaled_img, scaled_img.shape[0]//dimension)
    for rows in split_rows:
        split_columns = np.hsplit(rows, scaled_img.shape[1]//dimension) #this, along with the vsplit function above extracts grids from the array
        for cols in split_columns:
            all_blocks.append(cols) #individual blocks are stored as elements of a list 
    
    return all_blocks, n_rows, n_cols


#this function calculates the dct matrix (not the coefficients) according to the formula mentioned in
#the report
def calculate_dct_matrix(dimension):
    dct_matrix = np.empty((dimension, dimension), dtype = np.float32)
    for i in range(0, dimension):
        for j in range(0, dimension):
            if i == 0:
                dct_matrix[i, j] = 1/math.sqrt(dimension)   
            else:
                dct_matrix[i, j] = math.sqrt(2/dimension) * math.cos(((2*j+1)*i*math.pi)/(2*dimension))
    return dct_matrix


#this function calculates the dct coefficients based on the formula mentioned in the report
#it takes in the dct matrix and individual image blocks
def calculate_dct_coefficient(dct_matrix, image_block):
    dct_coeff = np.matmul(np.matmul(dct_matrix, image_block), np.transpose(dct_matrix, (1, 0)))
    return dct_coeff


#this function takes in the image's path, dimension of dct matrix, and number of partial coefficients.
#it returns all the original blocks in the image, partial set of dct coefficients, and the dct matrix,
#along with the number of rows and columns (needed for the next step)
def apply_dct_transform(img_pth, dimension, k):
    all_blocks, n_rows, n_cols = scale_and_split(img_pth, dimension) #scale and split the image
    dct_matrix = calculate_dct_matrix(dimension) #calculate dct matrix
    img = cv2.imread(img_pth, 0).astype(np.float32)
    all_dct_coeff = []
    priority_order = [1, 2, 9, 17, 10, 3, 4, 11, 18, 25, 33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 49, 42, 
                      35, 28, 21, 14, 7, 8, 15, 22, 29, 36, 43, 50, 57, 58, 51, 44, 37, 30, 23, 16, 24, 31, 
                      38, 45, 52, 59, 60, 53, 46, 39, 32, 40, 47, 54, 61, 62, 55, 48, 56, 63, 64] #zig-zag ordering
    rel_priority_order = priority_order[: k] #select the partial set
    masked_mat = np.empty(dct_matrix.shape) #masked matrix is a boolean matrix, where True indices are
                                            #selected in the dct coefficients matrix by multiplying the
                                            #coefficient matrix with this matrix.

    for row in range(masked_mat.shape[0]):
        for col in range(masked_mat.shape[1]):
            coeff_nbr = row * dimension + col + 1
            if coeff_nbr in rel_priority_order:
                masked_mat[row, col] = True #True if the index falls within the partial priority list
            else:
                masked_mat[row, col] = False #False if the index doesn't fall within the partial priority list

    for block_nbr, block in enumerate(all_blocks):
        dct_coeff = calculate_dct_coefficient(dct_matrix, block) #calculate dct coefficients for each block
        row_nbr = block_nbr // n_cols
        col_nbr = block_nbr - n_cols * row_nbr
        dct_coeff = dct_coeff * masked_mat #zero out the unnecessary coefficients (i.e. select only partial set)
        all_dct_coeff.append(dct_coeff) 
        
    return all_blocks, all_dct_coeff, dct_matrix, n_rows, n_cols


#this function reconstructs the image using the partial set of dct coefficients
def apply_inverse_dct_transform(img_pth, all_blocks, all_dct_coeff, dct_matrix, n_rows, n_cols, dimension):
    img = cv2.imread(img_pth, 0).astype(np.float32)
    transformed_img = np.empty(img.shape, dtype = np.float32) #placeholder new image
    for block_nbr, block in enumerate(all_blocks):
        row_nbr = block_nbr // n_cols
        col_nbr = block_nbr - n_cols * row_nbr
        transformed_block = np.matmul(np.matmul(np.transpose(dct_matrix, (1, 0)), all_dct_coeff[block_nbr]), dct_matrix) + 128 #reconstruction formula as mentioned in the report. 128 is added back to bring the image to range 0-255
        transformed_img[row_nbr*dimension:(row_nbr+1)*dimension, col_nbr*dimension:(col_nbr + 1)*dimension] = transformed_block #change the elements of the image block by block
    img = img.astype(np.uint8) #8 bit rgb image
    transformed_img = transformed_img.astype(np.uint8) #8 bit rgb image
    
    return img, transformed_img

#this function performs all the tasks in the project
if __name__ == '__main__':
    
    root_dir = '/kaggle/input/ece634-project2/sample_image'
    dimension = 8
    print("#######################################################")
    print("#######################################################")
    print("DCT TRANSFORM..... DCT TRANSFORM ...... DCT TRANSFORM")
    print("#######################################################")
    print("#######################################################")
    for img_name in ['pentagon.tif', 'goldhill.png', 'cameraman.tif']:
        img_pth = f'{root_dir}/{img_name}'
        for k in [64, 56, 48, 40, 32, 24, 16, 8]:
            save_pth = f'./img_name_{img_name}_k_{k}.jpg'
            all_blocks, all_dct_coeff, dct_matrix, n_rows, n_cols = apply_dct_transform(img_pth, dimension, k)
            old_img, new_img = apply_inverse_dct_transform(img_pth, all_blocks, all_dct_coeff, dct_matrix, n_rows, n_cols, dimension)
            psnr = 20 * math.log10(255) - 10 * math.log10(np.sum((new_img - old_img) ** 2)/(new_img.shape[0] * new_img.shape[1]))
            print(f"For img name = {img_name}, k = {k}, the PSNR is {psnr}")
            plt.imshow(new_img, cmap = 'gray')
            plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False) #Courtesy https://www.geeksforgeeks.org/how-to-remove-ticks-from-matplotlib-plots/
            plt.savefig(f'{save_pth}')
            plt.close()