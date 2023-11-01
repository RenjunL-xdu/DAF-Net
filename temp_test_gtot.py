from Evaluation.Test_GTOT import val_GTOT, get_exp
from Evaluation.Test_RGBT234 import val_RGBT234, get_txt
for num in range(169,170):
    path='/media/lx/Ubuntu_SSD/LRJ/DAFNet-master/runs/track/exp' + str(num)
    if not get_exp(num):
        continue
    val_GTOT([str(num) + ext for ext in ['_nobb', '_ori', '_mean', '_wbf', '_best']],path)
    # val_GTOT([str(num) + ext for ext in ['_best']],path)
