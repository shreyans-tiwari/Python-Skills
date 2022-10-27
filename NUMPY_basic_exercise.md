```python
# 20 Numpy Exercises for Beginners

```


```python
import numpy as np

print(np.__version__)
```

    1.21.5
    


```python
# Exercise 1 - Element-wise addition of 2 numpy arrays

a = np.array([[1,2,3],
             [4,5,6]])

b = np.array([[10,11,12],
            [13,14,15]])

c = a + b

print(c)
```

    [[11 13 15]
     [17 19 21]]
    


```python
# Exercise 2 -  Multiply a matrix (numpy array) by a scalar

a = np.array([[1,2,3],
             [4,5,6]])

b = 2 * a

print(b)
```

    [[ 2  4  6]
     [ 8 10 12]]
    


```python
# Exercise 3 - Identity Matrix

i = np.eye(4)

print(i)
```

    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    


```python
# Exercise 4 - Array re-dimensioning

a = np.array([x for x in range(27)])

o = a.reshape((3, 3, 3))

print(o)
```

    [[[ 0  1  2]
      [ 3  4  5]
      [ 6  7  8]]
    
     [[ 9 10 11]
      [12 13 14]
      [15 16 17]]
    
     [[18 19 20]
      [21 22 23]
      [24 25 26]]]
    


```python
# Exercise 5 - Array datatype conersion

a = np.array([[2.5, 3.8, 1.5],
             [4.7, 2.9, 1.56]])

o = a.astype('int')

print(o)
```

    [[2 3 1]
     [4 2 1]]
    


```python
# Exercise 6 - Obtaining Boolean Array from Binary Array

a = np.array([[1, 0, 0],
             [1, 1, 1],
             [0,0,0]])

o = a.astype('bool')

print(o)
```

    [[ True False False]
     [ True  True  True]
     [False False False]]
    


```python
# Exercise 7 - Horizontal Stacking of Numpy Arrays

a1 = np.array([[1, 2, 3],
              [4, 5, 6]])

a2 = np.array([[7, 8, 9],
              [10, 11, 12]])

o = np.hstack((a1, a2))

print(o)
```

    [[ 1  2  3  7  8  9]
     [ 4  5  6 10 11 12]]
    


```python
# Exercise 8 - Vertically Stacking Numpy Arrays

a1 = np.array([[1, 2],
              [3, 4],
              [5, 6]])

a2 = np.array([[7, 8],
              [9, 10],
              [10, 11]])

o = np.vstack((a1, a2))

print(o)
```

    [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [10 11]]
    


```python
# Exercise 9 - Custom Sequence Generation

list_of_numbers = [x for x in range(0, 101, 2)]

o = np.array(list_of_numbers)

print(o)
```

    [  0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34
      36  38  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
      72  74  76  78  80  82  84  86  88  90  92  94  96  98 100]
    


```python
# Exercise 10 - Getting the positions (indexes) where elements of 2 numpy arrays match

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 3, 2, 4, 5])

print(np.where(a == b))
```

    (array([0, 3, 4], dtype=int64),)
    


```python
# Exercise 11 - Generation of given count of equally spaced numbers within a specified region

o = np.linspace(0, 100, 5)

print(o)
```

    [  0.  25.  50.  75. 100.]
    


```python
# Exercise 12 - Matrix Generation with one particular value

o = np.full((2,3), 5)

print(o)

# Alternate Solution
a = np.ones((2,3))
o = 5 * a
print(o)
```

    [[5 5 5]
     [5 5 5]]
    [[5. 5. 5.]
     [5. 5. 5.]]
    


```python
# Exercise 13 - Array Generation by repeatition of a small array across each dimension

a = np.array([[1, 2, 3],
             [4, 5, 6]])

o = np.tile(a, 10)

print(o)
```

    [[1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3]
     [4 5 6 4 5 6 4 5 6 4 5 6 4 5 6 4 5 6 4 5 6 4 5 6 4 5 6 4 5 6]]
    


```python
# Exercise 14 - Array Generation of random integers within a specified range

np.random.seed(123) # Setting the seed

o = np.random.randint(0, 10, size = (5,5))

print(o)
```

    [[2 2 6 1 3]
     [9 6 1 0 1]
     [9 0 0 9 3]
     [4 0 0 4 1]
     [7 3 2 4 7]]
    


```python
# Exercise 15 - Array Generation of random numbers following normal distribution

np.random.seed(123)

o = np.random.normal(size = (3,3))

print(o)
```

    [[-1.0856306   0.99734545  0.2829785 ]
     [-1.50629471 -0.57860025  1.65143654]
     [-2.42667924 -0.42891263  1.26593626]]
    


```python
# Exercise 16 -  Matrix Multiplication

a = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])

b = np.array([[2,3,4],
            [5,6,7],
            [8,9,10]])

o = a@b

print(o)
```

    [[ 36  42  48]
     [ 81  96 111]
     [126 150 174]]
    


```python
# Alternate Solution

a = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])

b = np.array([[2,3,4],
             [5,6,7],
             [8,9,10]])

o = np.matmul(a, b)

print(o)
```

    [[ 36  42  48]
     [ 81  96 111]
     [126 150 174]]
    


```python
# Exercise 17 - Matrix Transpose

a = np.array([[1,2,3],
             [4,5,6],
             [7,8,9]])

a_transpose = a.T

print(a_transpose)
```

    [[1 4 7]
     [2 5 8]
     [3 6 9]]
    


```python
# Exercise 18 - Sine of Angel (in radians)

angles = np.array([3.14, 3.14/2, 6.28])

sine_of_angles = np.sin(angles)

print('Sine of the given array of angels = ', sine_of_angles)
```

    Sine of the given array of angels =  [ 0.00159265  0.99999968 -0.0031853 ]
    


```python
# Exercise 19 - Cosine of Angels (in radians)

angles = np.array([3.14, 3.14/2, 6.28])

cosine_of_angles = np.cos(angles)

print('Cosine of the given array of angles = ', cosine_of_angles)
```

    Cosine of the given array of angles =  [-9.99998732e-01  7.96326711e-04  9.99994927e-01]
    


```python
# Exercise 20 - Generating the array element indexes such that the array elements appear in ascending order

array = np.array([10, 1, 5, 2])

indexes = np.argsort(array)

print(indexes)
```

    [1 3 2 0]
    


```python

```
