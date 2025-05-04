import pycuda.autoinit
from pycuda.compiler import SourceModule
import os, sys

CUDA_FOLDER = "./CUDA"
 
def search_dir(path, files):
    files += [f for f in os.listdir(path) if os.path.isfile(path + os.sep + f) and f[-3:]==".cu"]
    for d in [path + os.sep + d for d in os.listdir(path) if os.path.isdir(path+os.sep+d) and d != "__pycache__"]:
        search_dir(d, files)
    

def init(files=None):
    """
   Generates SourceModule with the code and autoinits pycuda.

   :param [str] files: The files from which the code is generated (if none, all the files in the directory (recursive) are considered)
   :return: The SourceModule
   """
    if files == None:
        files = []
        search_dir(os.getcwd(), files)
    code = ""
    for file in files:
        code += open(os.path.join(CUDA_FOLDER, file), 'r').read() + '\n'
    return SourceModule(code)




if __name__ == "__main__":
    files = []
    search_dir(os.getcwd(), files)
    print(files)