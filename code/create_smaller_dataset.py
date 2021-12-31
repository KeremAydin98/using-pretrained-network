import os
import shutil

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def symlink(src,dst):
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)

#Changes directory
os.chdir("/home/kerem/Desktop/fruits-360/")

#Prints the current directory
print(os.getcwd())

mkdir("../fruits-360/fruits-smaller-dataset")

#Prints the filenames located in the current directory
print(os.listdir())

classes = [
    'Apple Golden 1',
    'Avocado',
    'Lemon',
    'Mango',
    'Kiwi',
    'Banana',
    'Strawberry',
    'Raspberry'
]

training_path_from = os.path.abspath('../fruits-360/Training')
test_path_from = os.path.abspath('../fruits-360/Test')

for i in classes:
    shutil.move(f"{training_path_from}/{i}/",f"../fruits-360/fruits-smaller-dataset/Training/{i}")
    shutil.move(f"{test_path_from}/{i}/",f"../fruits-360/fruits-smaller-dataset/Test/{i}/")

    symlink(f"{training_path_from}/", f"../fruits-360/fruits-smaller-dataset/Training/{i}/")
    symlink(f"{test_path_from}/", f"../fruits-360/fruits-smaller-dataset/Test/{i}/")