# Load CSV from URL using NumPy
from urllib.request import urlopen

from numpy import loadtxt

url = "https://goo.gl/bDdBiA"
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
print(dataset.shape)
