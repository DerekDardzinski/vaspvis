import numpy as np

up = np.ones((4,5)) * np.arange(5)
down = np.ones((4,5)) * np.flip(np.arange(5,10))
print(up)
print(down)

print(up-down)
print(up+down)
print((up-down) / (up+down))
