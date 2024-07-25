exclude = 5
k = 6
trains =(i for i in range(exclude)) + (i for i in range(exclude + 1, k))
print(trains)