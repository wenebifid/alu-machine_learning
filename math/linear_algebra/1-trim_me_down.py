#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
start = len(matrix[0]) // 2 - 1
the_middle = [[row[i] for i in range(start, start + 2)] for row in matrix]
print("The middle columns of the matrix are: {}".format(the_middle))
