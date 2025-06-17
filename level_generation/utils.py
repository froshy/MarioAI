import numpy as np
import os


def read_text_to_level(level):
    """
    Reads in a .txt file represenation of a mario level and creates a numpy array. 
    Returns a 2D array that simply places each line into an array, each contained in the large array
    
    e.g
    
    read_text_to_level(
        abcde
        fghij
        klmno
        )
    -->
    array([
        [a,b,c,d,e],
        [f,g,h,i,j],
        [k,l,m,n,o]
    ])
    
    Arguments:
        level (str or Path) - a path to the .txt file containing the leel
    
    Returns:
        2D np.array - the level as an array, each character is an element of an array, and each array is a row of the level (in a numpy array)
    """
    
    with open(level, 'r') as file:
        lines = [line.rstrip('\n') for line in file]
    
    level_arr = np.array([list(line) for line in lines])
    
    return level_arr

