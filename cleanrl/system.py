import os

directory = f'old_policy/'
lst = [i for i in range(1, 501)]
print(lst)

count = 0
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        print("File name:", filename)
    if count > 10:
        break
