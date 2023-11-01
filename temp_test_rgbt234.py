import os

from Evaluation.Test_RGBT234 import val_RGBT234, get_txt

for num in range(7, 8):
    path = '/media/lx/Ubuntu_SSD/LRJ/DAFNet-master/runs/track/exp' + str(num)
    if not os.path.exists(path):
        continue
    if not get_txt(num):
        print(f"exp{num} error!")
        continue
    val_RGBT234([str(num) + ext for ext in ['_nobb', '_ori', '_mean', '_wbf', '_best']], path)
    # val_RGBT234([str(num) + ext for ext in ['_nobb', '', '_best']], path)
