a=[1,2,3,4,5]
b=[3,3,3,3,3]
import numpy as np

a=np.asarray(a)
b=np.asarray(b)

print(((a-b)>0).astype(np.uint8))