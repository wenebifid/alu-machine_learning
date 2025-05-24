#!/usr/bin/env python3
"""
This module provides a function to determine the shape of a matrix.
"""


def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list of dimensions.
    Args:
        matrix (list): A nested list representing the matrix.
    Returns:
        list: A list of integers representing the dimensions of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
