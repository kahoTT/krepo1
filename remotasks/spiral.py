# Here is a Python function that prints a 2D matrix.
# ```python
def print_spiral(matrix):
    """
    This function takes a 2D matrix as input and prints the elements in a spiral order.

    The approach is to maintain four variables: top, bottom, left, and right to denote the boundaries
    of the matrix. We traverse the matrix in a spiral manner by updating these boundaries after
    each complete row or column traversal.

    :param matrix: List of Lists containing integers.
    :return: None
    """
    if not matrix:
        print("Matrix is empty.")
        return
    
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse from left to right along the top row
        for i in range(left, right + 1):
            print(matrix[top][i], end=" ")
        top += 1

        # Traverse from top to bottom along the right column
        for i in range(top, bottom + 1):
            print(matrix[i][right], end=" ")
        right -= 1

        # Traverse from right to left along the bottom row (only if top <= bottom)
        if top <= bottom:
            for i in range(right, left -1, -1):
                print(matrix[bottom][i], end=" ")
            bottom -= 1

        # Traverse from bottom to top along the left column (only if left <= right)
        if left <= right:
            for i in range(bottom, top - 1, -1):
                print(matrix[i][left], end=" ")
            left += 1

    print()

# Example usage
# matrix = [
#     [1,  2,  3,  4],
#     [5,  6,  7,  8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ]
# 
# print_spiral(matrix)
# 
# ```
# 
# In this program, the print_spiral function takes a matrix (a list of lists) as an input, and it prints the elements in a spiral order. The spiral order is achieved by maintaining four boundaries (top, bottom, left, and right) and traversing along the edges of the matrix, updating these boundaries after each complete row or column traversal.
# 
# Please ensure that the input matrix is not empty before calling the function, as the function does not handle this case.
# 