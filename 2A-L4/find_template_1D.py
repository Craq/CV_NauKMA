import numpy as np
import scipy.signal as sp


def find_template_1D(t: np.array, s: np.array) -> int:
    # Locate template t in signal s and return index. Use scipy.signal.correlate2d
    corr = sp.correlate2d(s, t)
    return np.argmax(corr) - t.shape[1] + 1

s = np.array([[-1, 0, 0, 5, 1, 1, 0, 0, -1, -7, 2, 1, 0, 0, -1]])
t = np.array([[-1, -7, 2]])

print("Signal: \n {} \n {}".format(np.array(range(s.shape[1])), s[0]))
print("Template: \n {} \n {}".format(np.array(range(t.shape[1])), t[0]))

index = find_template_1D(t, s)
print(f"Index: s[0, {index}:{index+t.shape[1]}]")