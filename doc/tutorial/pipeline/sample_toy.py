import os
import random
import shutil
import glob
from tqdm import tqdm

random.seed(10)

destination = 'fate/examples/data/toy_cifar10'
if not os.path.exists(os.path.join(destination, 'train')):
    os.makedirs(os.path.join(destination, 'train'))

if not os.path.exists(os.path.join(destination, 'test')):
    os.makedirs(os.path.join(destination, 'test'))
    
shutil.copy('fate/examples/data/cifar10/labels.txt', os.path.join(destination, 'labels.txt'))


train_data_path = "fate/examples/data/cifar10/train"
test_data_path = "fate/examples/data/cifar10/test"

num_train_sample = 500
num_test_sample = 100


# Train toy
for dirname in tqdm(os.listdir(train_data_path)):
    
    files = glob.glob(os.path.join(train_data_path, dirname, "*.png"))
    toy_samples = random.sample(files, num_train_sample)
    
    if not os.path.exists(os.path.join(destination, 'train', dirname)):
        os.makedirs(os.path.join(destination, 'train', dirname))
        
        for toy_sample in toy_samples:
            shutil.copy(toy_sample, os.path.join(destination, 'train', dirname, toy_sample.split('/')[-1]))
        
        
# Test toy
for dirname in tqdm(os.listdir(test_data_path)):
    
    files = glob.glob(os.path.join(test_data_path, dirname, "*.png"))
    toy_samples = random.sample(files, num_test_sample)
    
    if not os.path.exists(os.path.join(destination, 'test', dirname)):
        os.makedirs(os.path.join(destination, 'test', dirname))
        
        for toy_sample in toy_samples:
            shutil.copy(toy_sample, os.path.join(destination, 'test', dirname, toy_sample.split('/')[-1]))
