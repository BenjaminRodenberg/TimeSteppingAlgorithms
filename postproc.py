import numpy as np
import argparse

import matplotlib.pyplot as plt
import itertools
import json, glob
from numpy import nan as NAN

parser = argparse.ArgumentParser(description='Postprocessing')
parser.add_argument('foldername', type=str, help='enter folder with experimental data here')

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['lmodern'], 'size'   : 15})
plt.rc('text', usetex=True)

legends = []
lines = []
name = parser.parse_args().foldername
plottingdir = r'./' + name
filenames = glob.glob(plottingdir+"/*.json")
print("reading from")
print(filenames)

marker_collection = ('o','x','d','s','^','<','>','v','.')

marker_left = itertools.cycle(marker_collection)

plot_right = True
if plot_right:
    marker_right = itertools.cycle(marker_collection)

if plot_right:
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12,9))
else:
    f, (ax1) = plt.subplots(1, 1, sharey=True, figsize=(12,9))

data_left = np.empty(shape=(0,0))
data_right = np.empty(shape=(0,0))
data_header = []


def concat_w_zero_padding(a,b):
    if len(a.shape) > 1 and a.shape[1] == 0:  # a has no entries
        return b
    elif len(b.shape) > 1 and b.shape[1] == 0:  # b has no entries
        return a
    else:  # both arrays have entries
        la = a.shape[0]
        lb = b.shape[0]
        if la < lb:  # padding a
            a = np.pad(a, (0, lb-la), 'constant')
        elif la > lb:  # padding b
            b = np.pad(b, (0, la-lb), 'constant')
        else:  # no padding needed if la = lb
            pass
        return np.c_[a,b]


for filename in filenames:
    print("---")
    print(filename)
    with open(filename) as data_file:

        data = json.load(data_file)
        taus = np.array(data['temporal_resolution'])

        if data.has_key('errors_left'):
            errors_left = np.array(data['errors_left'])
            errors_left[errors_left > 10.0] = NAN
            ax1.dataLim._set_x1(taus.max())
            line = ax1.loglog(taus, errors_left, ':',marker=marker_left.next(),markerfacecolor="none",markersize=10,markeredgewidth=2)
            data_left = concat_w_zero_padding(data_left, errors_left)
        elif data.has_key('errors'):
            errors = np.array(data['errors'])
            errors[errors > 10.0] = NAN
            line = ax1.loglog(taus, errors, marker=marker_left.next(),markerfacecolor="none")
            data_left = concat_w_zero_padding(data_left, errors)

        if plot_right and data.has_key('errors_right'):
            errors_right = np.array(data['errors_right'])
            errors_right[errors_right > 10.0] = NAN
            ax2.loglog(taus, errors_right, ':',marker=marker_right.next(),markerfacecolor="none",markersize=10,markeredgewidth=2)
            data_right = concat_w_zero_padding(data_right, errors_right)
        elif plot_right and data.has_key('errors'):
            errors = np.array(data['errors'])
            errors[errors > 10.0] = NAN
            ax2.loglog(taus, errors, marker=marker_right.next())
            data_right = concat_w_zero_padding(data_right, errors)

        if data['numerical_parameters'].has_key('neumann coupling scheme'):
            legends.append(data['experiment_name'])
        elif data['numerical_parameters'].has_key('neumann coupling order'):
            legends.append(data['experiment_name']+" - FD order: "+ str(data['numerical_parameters']['neumann coupling order']))
        else:
            legends.append(data['experiment_name'])

        data_header.append(data['experiment_name'])

        lines.append(line[0])

data_left = concat_w_zero_padding(data_left, taus)
data_right = concat_w_zero_padding(data_right, taus)
data_header.append('taus')

def plot_order_line(order, ax, x_min, x_max, y_min, y_max):
    x = np.zeros(2)
    y = np.zeros(2)

    size = 10.0
    x[0], y[0] = x_max, y_min * size**order * .1
    x[1], y[1] = 1.0/size*x_max, y_min * .1
    x *= .5
    legend = r" $\mathcal{O}(\tau^{"+str(order)+"})$"
    ax.loglog(x,y,'k')
    ax.annotate(legend, xy=(x[0]*1.1, y[0]), horizontalalignment='left', verticalalignment='center')
    return

ax1.grid()
ax1.set_title(r"error in left domain $\Omega_L$")
ax1.set_xlabel(r"time step $\tau$")
ax1.set_ylabel(r"error $\epsilon$")
x_min, y_min = ax1.dataLim._get_min()
x_max, y_max = ax1.dataLim._get_max()
plot_order_line(1, ax1, x_min, x_max, y_min, y_max)
plot_order_line(2, ax1, x_min, x_max, y_min, y_max)
plot_order_line(4, ax1, x_min, x_max, y_min, y_max)

if plot_right:
    x_min, y_min = ax2.dataLim._get_min()
    x_max, y_max = ax2.dataLim._get_max()
    plot_order_line(1, ax2, x_min, x_max, y_min, y_max)
    plot_order_line(2, ax2, x_min, x_max, y_min, y_max)
    plot_order_line(4, ax2, x_min, x_max, y_min, y_max)

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.8])

if plot_right:
    ax2.grid()
    ax2.set_title(r"error in right domain $\Omega_R$")
    ax2.set_xlabel(r"time step $\tau$")
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

f.legend(lines, legends, 'lower center',bbox_to_anchor=(0.5, 0.05),prop={'size': 15})
f.set_size_inches(6,6)
plt.subplots_adjust(left=0.15, bottom=0.4, right=0.90, top=0.93, wspace=None, hspace=None)
plt.savefig(plottingdir + '//' + name+'.pdf')
np.savetxt(plottingdir + '//data_left.csv', data_left, header=', '.join(data_header),delimiter=',')
np.savetxt(plottingdir + '//data_right.csv', data_right, header=', '.join(data_header),delimiter=',')
plt.show()
