import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import cv2 as cv
import nltk
import sklearn 
import re
import os

#22.10.2025 Модуль numpy: работа с массивами, математические операции

# ls = [1, [2,5], [9,10,12]]  #список Питона
# print(type(ls))  #<class 'list'>
# arr = [[1,2,3],[4,5,6],[7,8,9]]  #однородный массив-list like NumPy

#Задача 1. Создайте одномерный массив из чисел от 0 до 9 включительно. Выведите его на экран.
'''
arr = np.arange(9) #start=0:stop:step=1
print(type(arr))   #<class 'numpy.ndarray'>
print(arr)         #[0 1 2 3 4 5 6 7 8]   numpay array

a = np.arange(2,8,2) #start=2:stop=8:step=2
print(a)  #[2 4 6]

b = np.array([0,1,2,3,4,5,6,7,8,9])   # list => ndarray
print(type(b))  #<class 'numpy.ndarray'>
print(b)        #[0 1 2 3 4 5 6 7 8 9]

lns1 = np.linspace(0,100,101) #start,stop,num_points
print(type(lns1))  #<class 'numpy.ndarray'>
print(lns1)        #[  0.   1.   2.   3.     98.  99. 100.]

lns2 = np.linspace(10,15,6)
print(lns2)     #[10. 11. 12. 13. 14. 15.]
print(lns2[0])          #10.0
print(type(lns2[0]))    #<class 'numpy.float64'>
'''

#Задача 2. Создайте двумерный массив 3×4, заполненный нулями. 
#Затем замените все элементы второго столбца на единицы.
'''
arr2 = np.array([[0,1,2,3,4], [5,6,7,8,9]])   #не то
print(type(arr2)) #<class 'numpy.ndarray'>
print(arr2)       #[[0 1 2 3 4]
                  # [5 6 7 8 9]]

arr3 = np.zeros((3,4))  # rows=3, columns=4
print(type(arr3))
print(arr3)

# <class 'numpy.ndarray'>
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]:

print(arr3[0])      #[0. 0. 0. 0.]

# arr3[0,1] = 1
# arr3[1,1] = 1 
# arr3[2,1] = 1

# arr3[:,1] = np.array([1,1,1])
# arr3[:,1] = [1,1,1]
arr3[:,1] = 1

print(arr3)
# [[0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 1. 0. 0.]]

print(arr3[:,0])   #[0. 0. 0.]  столбец 0
print(arr3[:,1])   #[1. 1. 1.]  столбец 1

print(arr3[0][1])   #1.0
print(arr3[0,1])    #1.0
'''
# Задача 3. Сгенерируйте массив 5×5 со случайными числами от 0 до 1, используя np.random.rand. 
# Найдите максимальный и минимальный элемент массива, а также их индексы

