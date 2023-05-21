import os
import re

directory = f'policy/'
# lst = [i for i in range(1, 501)]
# print(lst)
epsilon = 0

lst = []
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        e_match = re.search(f'epsilon_{epsilon}_', filename)
        if e_match:
            p_match = re.search(r'performance_(\d+)_', filename)
            lst.append(int(p_match[1]))

lst.sort()
print(lst)
print(len(lst))