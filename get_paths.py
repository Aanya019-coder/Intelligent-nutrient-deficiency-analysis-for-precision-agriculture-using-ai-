import kagglehub
import os

p1 = kagglehub.dataset_download('emmarex/plantdisease')
p2 = kagglehub.dataset_download('guy007/nutrientdeficiencysymptomsinrice')

with open('paths.txt', 'w') as f:
    f.write(p1 + '\n')
    f.write(p2 + '\n')
