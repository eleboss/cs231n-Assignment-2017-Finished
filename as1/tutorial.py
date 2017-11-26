def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)


print(quicksort([0, 1, 5, 4, 7, 1999, 5, 5, 5, 4, 4, 5]))
x = 3
print(type(x))
print(x)
print(x + 1)
print(x - 1)
print(x * 2)
print(x ** 2)
x += 1
print(x)
x *= 2
print(x)
y = 11
y = y % 2
print(type(y))
print(y, y + 1, y * 2, y ** 2)

t = True
f = False
print(type(t), type(f))
print(t and f, t or f, not t, t != f, )
hello = "hello"
world = 'world'
print(hello, world, len(hello))
hw = hello + ' ' + world
print(hw)
hw12 = '%s %s %d' % (hello, world, 12)
print(hw12)
s = 'hello'
print(s.capitalize())
print(s.center(100))
print(s.replace('o', '(ell)'), s.replace('l', '(ell)'))
print(' world'.strip())
print(s.replace('l', '(ell)'))

xs = [3, 1, 2]
print(xs, xs[2])
xs[2] = 'foo'
print(xs)
xs.append("bar")
print(xs)
x = xs.pop()
print(x, xs)
xs1 = [1, 3, 3, 32, 1, 3]
print(xs1)
xs1.sort()
print(xs1)

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
for idx, animal in enumerate(animals):
    print('#%d: %s while the orign idx is %d' % (idx + 1, animal, idx))

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
    print(squares)
print(nums[:-1])
nums[2:4] = [8, 9]
print(nums)
squares = [x ** 2 for x in nums]
print(squares)
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)

# dictionary
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat'])
print('cat' in d)
d['fish'] = 'wet'
print(d['fish'])
print(d.get('monkey', 'N/A'))
print(d.get('fish', 'N/A'))
del d['fish']
print(d.get('fish', 'N/A'))

# loops
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

# Dictionary comprehensions
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

# Set
animals = {'cat', 'dog'}
print('cat' in animals)
print('fish' in animals)
animals.add('fish')
print('fish' in animals)
print(len(animals))
animals.add('cat')
print(len(animals))
animals.remove('cat')
print(len(animals))

# Loop in set
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

from math import sqrt
nums = {int(sqrt(x)) for x in range(37)}
print(nums)

# Tuples
d = {(x, x + 1): x for x in range(10)}
print(d)
t = (5, 6)
print(type(t))
print(d[t])
print(d[(2, 3)])

# Functions
def sign(x):
    if x > 0:
        return 'postive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'


for x in [-1, 0, 10]:
    print(sign(x))

def hello(name, loud):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' %name)

hello('bob', False)

# Classes
class Greeter(object):

    def __init__(self, name, id):
        self.name = name
        self.id = id
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred', 13)
print(g.id)
g.greet()
g.greet(loud=True)

# numpy Arrays
import numpy as np


a = np.array([1,2,3])
print(type(a))
print(a.shape)
print(a[0],a[1],a[2])
a[0]=5
print(a)

b = np.array([[1,2,3], [4,5,6]])
print(b)
print(b.shape)
print(b[1,1])

c = np.zeros((2,2))
print(c)
d = np.ones((1,2))
print(d)
e = np.full((2,2),7)
print(e)
f = np.random.random((2,2))
print(f)

# Array indexing
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[0:2, 0:3]
print(a[1,1])
print(a,b)
print('1231321321')
row_r1 = a[1,:]
print(row_r1)
row_r2 = a[1:2, :]
print(row_r2, row_r2.shape)
print(row_r2[0,1])
row_r2[0,1] = 100
print(row_r1[0])

# integer array indexing
a = np.array([[1,2], [3,4], [5,6]])
print(a,a[[0,1,2], [0,1,0]])
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
print(a[[0,0],[1,1]])
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])
print(np.arange(4))
a[np.arange(4), b] += 10
print(a)

# boolean array indexing
a = np.array([[1,2], [3,4], [5,6]])
bool_idx=(a > 2)
print(bool_idx)
print(a[bool_idx])
print(a[a>2])

# Datatype
x = np.array([1,2])
print(x.dtype)

x = np.array([1.0,2.0])
print(x.dtype)

x = np.array([1,2], dtype=np.int64)
print(x.dtype)

# Array math
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print("1")
print(x+y)
print(np.add(x,y))

print(x - y)
print(np.subtract(x,y))

print(x*y)
print(np.multiply(x,y))

print(x / y)
print(np.divide(x,y))
print(np.sqrt(x))

print(x.dot(y))
print(np.dot(x,y))

print(np.sum(x))
print(np.sum(x, axis= 0))
print(np.sum(x, axis=1))

print(x)
print(x.T)

v = np.array([1,2,3])
print(v)
print(v.T)
print('nwxt')
# Broadcasting
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1,0,1])
y = np.empty_like(x)
print(y)

for i in range(4):
    y[i, :] =x[i,:] +v
print(y)
vv = np.tile(v,(4,1))
print(vv)

from numpy import mat,matrix

v = mat([1,2,3])
w = mat([4,5])
print(v , w)
v = np.reshape(v, (3,1))
print(v)
print(np.dot(v,w))


# scipy
from scipy.misc import imread, imsave, imresize

img = imread('house.jpg')
print(img.dtype, img.shape)
img_tinted = img * [1, 0.95 ,0.9]
img_tinted = imresize(img_tinted, (300,300))

imsave('house_tinted.png', img_tinted)

from scipy.spatial.distance import pdist, squareform
x = np.array([[0,1], [1,0], [2,0]])
print(x)
d = squareform(pdist(x, 'euclidean'))
print(d)

# Matplatlib
import matplotlib.pyplot as plt

x = np.arange(0, 3 *np.pi, 0.1)
print(x)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(4,1,1)

plt.plot(x,y_sin)
plt.subplot(4,1,2)
plt.plot(x,y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Since and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# Images
img = imread('house.jpg')
img_tinted = img * [1, 0.95, 0.9]

plt.subplot(4,2,1)
plt.imshow(img)
plt.subplot(4, 2, 2)
plt.imshow(np.uint8(img_tinted))
plt.show()