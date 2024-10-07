import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rcParams.update({'font.size':20})
fig, ax = plt.subplots()
ax.grid(True)
ax.set_ylim(-0.25,2.25)
ax.set_xlim(-0.25,2.25)
plt.xticks(np.arange(0, 3, 1), ["xᵢ₋₁ ", "xᵢ", "xᵢ₊₁"])
plt.yticks(np.arange(0, 3, 1), ["tⱼ₋₁", "tⱼ", "tⱼ₊₁"])
plt.scatter([1,0,1,2,1],[0,1,1,1,2], c="red", s=100)


plt.show()