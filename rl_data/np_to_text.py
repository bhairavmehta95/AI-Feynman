import numpy as np

filename = 'Reacher3DOF-v0-forwardkinematics'

a = np.load(f'{filename}.npz')

xcoord_data = []
ycoord_data = []
for state, label in zip(a['states'], a['labels']):
    xcoord_data.append(np.concatenate((state, [label[0]] )))
    ycoord_data.append(np.concatenate((state, [label[1]] )))

with open(f'{filename}_y.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % ' '.join(str(x) for x in element) for element in ycoord_data)