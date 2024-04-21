import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

SIZE = float(sys.argv[1])
N = int(sys.argv[2])
DT = float(sys.argv[3])
TEMP = float(sys.argv[4])
DISTR = sys.argv[5]
INTEG = sys.argv[6]
TIME_SPAN = float(sys.argv[7])
RECTIME = float(sys.argv[8])
RECPOINTS = int(sys.argv[9])
DX = SIZE/(N-1)

def phi_s(x, x0=0):
    return np.sqrt(2)/np.cosh(x-x0-SIZE/2)

x = np.linspace(0, SIZE, N)
t = np.linspace(0, RECTIME, RECPOINTS+1)

Phi_av = np.loadtxt('Out/Phi_av_T='+f'{TEMP:.6f}'+'.txt')
E_pot_long_av = np.loadtxt('Out/E_pot_long_av_T='+f'{TEMP:.6f}'+'.txt')

i_sph = np.argmax(E_pot_long_av)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""        Animation: field evolution near the barrier         """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

width=1280
height=1024

vs = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('out1.avi', fourcc, 15.0, (width, height))

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.title('Thermal Activation at '+r'$T=0.1,\eta=0$',x=-0.1,y=1.2)
ax1.set_box_aspect(0.75)
ax2.set_box_aspect(0.75)

UU = np.zeros(len(E_pot_long_av)-10)
for i in range(len(UU)):
    UU[i] = E_pot_long_av[-10-i]

ax1.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=10)
ax1.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax1.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax1.set_xlabel(r'$x$',fontsize=10)
ax1.set_ylabel(r'$\varphi$',fontsize=10,rotation='horizontal')
ax1.yaxis.set_label_coords(-0.02,1.04)
ax1.set_ylim([-0.6,2.9])
ax1.set_xlim([-5.9,5.9])

ax2.plot(t[0:91], UU,'black',linewidth=1)
ax2.plot(t[len(t)-10-i_sph],E_pot_long_av[i_sph],'ro--')
ax2.tick_params(axis='both',which='both',right=True,top=True,direction='in',labelsize=10)
ax2.xaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax2.yaxis.grid(True,linestyle='dashed',linewidth=0.5,color='grey')
ax2.set_xlabel(r'$t$',fontsize=10)
ax2.set_ylabel(r'$\langle U(t)\rangle$',fontsize=10,rotation='horizontal')
ax2.yaxis.set_label_coords(-0.02,1.01)
ax2.set_xlim([0,5])
ax2.set_ylim([-2.9,2.4])

labelT = ax1.text(-5,2.5,f't={t[0]:.2f}',fontsize=9)
ax1.plot(x-50,phi_s(x),'red',linestyle='dashed')
line1, = ax1.plot(x-50, Phi_av[-10], 'black', lw=1)

line2, = ax2.plot(t[0],E_pot_long_av[-10],'ko--')
i = 0
while True:
    i += 1
    frame = vs.read()[1]
    line1.set_ydata(Phi_av[-10-i])
    labelT.set_text(f't={t[i]:.2f}')
    line2.set_xdata(t[i])
    line2.set_ydata(E_pot_long_av[-10-i])
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))

    frame = cv.resize(frame, (width, height))
    data = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    data = cv.resize(data, (frame.shape[1], frame.shape[0]))

    out.write(data)
    cv.imshow('Stream', data)

    if len(Phi_av)-i-11 < 0:
        break

vs.release()
out.release()
cv.destroyAllWindows()
cv.waitKey(1)