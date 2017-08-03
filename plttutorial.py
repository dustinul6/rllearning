import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0., 5., 0.2)
plt.plot(t,t,'b--', t,t*2, 'go', t, t**2, 'r^')
plt.ylabel('sume numbers')
plt.show()

