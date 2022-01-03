import math

import numpy as np

b = 5.2

output = np.log(b)

print(np.log(b))
print(math.e ** output)

softmax = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax[0] * target_output[0] +
                  softmax[1] * target_output[1] +
                  softmax[2] * target_output[2]))

print(loss)


#bostonHousing