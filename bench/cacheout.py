# Program to clean out the filesystem cache
import numarray

a=numarray.arange(1000*1000*25)  # 100 MB of RAM
b=a*3  # Another 100 MB
# delete the reference to the booked memory
del a
del b

# Do a loop to fully recharge the python interpreter
j = 2
for i in range(1000*1000):
    j+=i*2

