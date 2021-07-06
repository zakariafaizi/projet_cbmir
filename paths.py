from typing import List
from os import listdir

path = "C:/Users/Zakaria/Documents/DECSession6/Stage/CBMIR/Images/"



outex_path = path+"Outex/"
outex_dir:List[str] = listdir(outex_path)

queryimg_path = path+"QueryImage/"
queryimg_dir:List[str] = listdir(queryimg_path)

