import numpy as np

my_list_array = [1, 4, 3, 4, 5]

numpy_list = np.array(my_list_array)
print(numpy_list)
multi_list = numpy_list[0] * 10
print(multi_list)
tuple1 = (1, 2, "j", 3, 4, "Hi there")
print(tuple1[2])
numpy_tuple = np.array(tuple1)
print(numpy_tuple)


def Getsum(myList):
    sum = 0
    for i in myList:
        sum = sum + i

lis = [2, 4, 6]
print(Getsum(lis))
