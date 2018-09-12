import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('../bg_srcs_noRepeater.pkl') as f: srcs = pickle.load(f)
ra = [x if x < np.pi else x-2*np.pi for x in srcs['RA']]
dec = srcs['Dec']

rcolor = 'darkmagenta'
new, old = '#e06000', '#40a0a0' 
colors = np.array(['#000000']*len(ra))

for i in range(len(ra)):
  if '130628' in srcs['FRB'][i]: 
    colors[i]=old
  elif int(srcs['FRB'][i][3:]) > 140600:
    if '150418' in srcs['FRB'][i]: 
      colors[i]=old
    else: colors[i]=new
  else: colors[i]=old

  if '110220' in srcs['FRB'][i]: 
    ra[i]+=0.01; dec[i]-=0.01
  if '140514' in srcs['FRB'][i]: 
    ra[i]-=0.01; dec[i]+=0.01

sizes = [6 if c==new else 5 for c in colors]

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, projection='mollweide')

ax.plot((18.+14/60.)/24.*2*np.pi-2*np.pi, (33.+5/60.)*np.pi/180, 
        mfc=rcolor, alpha=1, ms=7, mec='k', marker='^', 
        linestyle='', label='FRB 121102')
for i in range(len(ra)): 
  if i == 0:
    ax.plot(ra[i], dec[i], linestyle='', label='new in L2 analysis',
            mfc=colors[i], alpha=1, ms=6, mec='k', marker='o')
  if i == len(ra)-1:
    ax.plot(ra[i], dec[i], linestyle='', label='used in 6yr analysis',
            mfc=colors[i], alpha=1, ms=5, mec='k', marker='o')
  else:
    ax.plot(ra[i], dec[i], linestyle='',
            mfc=colors[i], alpha=1, ms=sizes[i], mec='k', marker='o')
ax.grid(True)

ax.legend()

plt.savefig('/home/sfahey/public_html/FRB/Analysis_L2/skymap.pdf')
plt.savefig('/home/sfahey/public_html/FRB/Analysis_L2/skymap.png')
