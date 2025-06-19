import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import os


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def list_add(x, y):
    return [a+0.6*b for a,b in zip(x, y)]

# each one include
# [fast_perceived, fast_full, slow_perceived, slow_full]
# datasets = ["laion", "sift", "trip", "msmarco", "laion", "sift", "trip", "msmarco"]
def draw_latency_breakdown(raw_data):

    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': "Times New Roman"})

    color1 = "#e6a002"
    color2 = "#019e73"

    error_style = {
        'elinewidth': 0.7,       # Width of error bar line
        'ecolor': 'dimgray',       # Color of error bars
        'capsize': 3,          # Length of the error bar caps
        'capthick': 0.7,         # Thickness of the caps
        'linestyle': '--'      # Style of error bar line
    }

    # Data
    categories1 = ['LAION', 'SIFT1M']

    perceived_fast_sh = [raw_data[0][0].mean(), raw_data[1][0].mean(), raw_data[2][0].mean(), raw_data[3][0].mean()]
    full_fast_sh = [raw_data[0][1].mean(), raw_data[1][1].mean(), raw_data[2][1].mean(), raw_data[3][1].mean()]
    evict_fast_sh = [x - y for x, y in zip(full_fast_sh, perceived_fast_sh)]
    perceived_err_fast_sh = [np.std(raw_data[0][0]), np.std(raw_data[1][0]), np.std(raw_data[2][0]), np.std(raw_data[3][0])]
    full_err_fast_sh = [np.std(raw_data[0][1]), np.std(raw_data[1][1]), np.std(raw_data[2][1]), np.std(raw_data[3][1])]


    perceived_fast_mal = [raw_data[4][0].mean(), raw_data[5][0].mean(), raw_data[6][0].mean(), raw_data[7][0].mean()]
    full_fast_mal = [raw_data[4][1].mean(), raw_data[5][1].mean(), raw_data[6][1].mean(), raw_data[7][1].mean()]
    evict_fast_mal = [x - y for x, y in zip(full_fast_mal, perceived_fast_mal)]
    perceived_err_fast_mal = [np.std(raw_data[4][0]), np.std(raw_data[5][0]), np.std(raw_data[6][0]), np.std(raw_data[7][0])]
    full_err_fast_mal = [np.std(raw_data[4][1]), np.std(raw_data[5][1]), np.std(raw_data[6][1]), np.std(raw_data[7][1])]

    perceived_slow_sh = [raw_data[0][2].mean(), raw_data[1][2].mean(), raw_data[2][2].mean(), raw_data[3][2].mean()]
    full_slow_sh = [raw_data[0][3].mean(), raw_data[1][3].mean(), raw_data[2][3].mean(), raw_data[3][3].mean()]
    evict_slow_sh = [x - y for x, y in zip(full_slow_sh, perceived_slow_sh)]
    perceived_err_slow_sh = [np.std(raw_data[0][2]), np.std(raw_data[1][2]), np.std(raw_data[2][2]), np.std(raw_data[3][2])]
    full_err_slow_sh = [np.std(raw_data[0][3]), np.std(raw_data[1][3]), np.std(raw_data[2][3]), np.std(raw_data[3][3])]

    perceived_slow_mal = [raw_data[4][2].mean(), raw_data[5][2].mean(), raw_data[6][2].mean(), raw_data[7][2].mean()]
    full_slow_mal = [raw_data[4][3].mean(), raw_data[5][3].mean(), raw_data[6][3].mean(), raw_data[7][3].mean()]
    evict_slow_mal = [x - y for x, y in zip(full_slow_mal, perceived_slow_mal)]
    perceived_err_slow_mal = [np.std(raw_data[4][2]), np.std(raw_data[5][2]), np.std(raw_data[6][2]), np.std(raw_data[7][2])]
    full_err_slow_mal = [np.std(raw_data[4][3]), np.std(raw_data[5][3]), np.std(raw_data[6][3]), np.std(raw_data[7][3])]

    # todo: compute diff and error bar


    # perceived_fast_sh = [0.024, 0.027, 0.139, 0.197]
    # evict_fast_sh = [0.028, 0.028, 0.29, 0.511]

    # perceived_fast_mal = [0.027, 0.033, 0.164, 0.233]
    # evict_fast_mal = [0.06, 0.076, 0.545, 0.964]

    # perceived_slow_sh = [0.581, 0.585, 0.985, 1.236]
    # evict_slow_sh = [0.277, 0.275, 3.19, 5.32]

    # perceived_slow_mal = [0.622, 0.595, 1.172, 1.251]
    # evict_slow_mal = [0.4, 0.414, 3.466, 5.752]

    # error_kw={
    #     'ecolor': 'black',           # color of the error bar
    #     'elinewidth': 2,             # width of error bar lines
    #     'capsize': 1,                # size of the cap at the end of bars
    #     'capthick': 1,               # thickness of the cap line
    #     'linestyle': '--',           # error bar line style
    #     'alpha': 0.7,                # transparency of error bars
    #     'lolims': ...,               # draw only lower limit error bars
    #     'uplims': ...,               # draw only upper limit error bars
    # }


    bar1_subcat1 = [3, 7]  # First bar for Category 1 and Category 2
    bar2_subcat1 = [4, 6]  # Second bar for Category 1 and Category 2

    categories2 = ['TripClick', 'MS MARCO']
    bar1_subcat2 = [2, 3]  # First bar for Category 3 and Category 4
    bar2_subcat2 = [5, 6]  # Second bar for Category 3 and Category 4

    # X-axis positions
    x1 = np.arange(len(categories1))
    x2 = np.arange(len(categories2))

    # Width of the bars
    bar_width = 0.35

    # Create the figure and two subplots
    fig, axs = plt.subplots(2, 2, figsize=(6, 3.8))

    ax1 = axs[0][0]
    ax2 = axs[0][1]
    ax3 = axs[1][0]
    ax4 = axs[1][1]

    # First subplot
    ax1.bar(x1 - bar_width/2, perceived_fast_sh[0:2], bar_width, yerr=perceived_err_fast_sh[0:2], error_kw=error_style, color=color1, label='Perceived Latency - SH.')
    ax1.bar(x1 - bar_width/2, evict_fast_sh[0:2], bar_width, yerr=full_err_fast_sh[0:2], error_kw=error_style, bottom=perceived_fast_sh[0:2], color=color1, alpha=0.5, label='Full Latency - SH.')

    ax1.bar(x1 + bar_width/2, perceived_fast_mal[0:2], bar_width, yerr=perceived_err_fast_mal[0:2], error_kw=error_style, color=color2, label='Perceived Latency - Mal.')
    ax1.bar(x1 + bar_width/2, evict_fast_mal[0:2], bar_width, yerr=full_err_fast_mal[0:2], error_kw=error_style, bottom=perceived_fast_mal[0:2], color=color2, alpha=0.5, label='Full Latency - Mal.')

    perceived_fast = [perceived_fast_sh, perceived_fast_mal]
    evict_fast = [evict_fast_sh, evict_fast_mal]

    perceived_fast_pos = [list_add(perceived_fast_sh, perceived_err_fast_sh), list_add(perceived_fast_mal, perceived_err_fast_mal)]
    evict_fast_pos = [list_add(evict_fast_sh, full_err_fast_sh), list_add(evict_fast_mal, full_err_fast_mal)]

    xs = [- bar_width/2, + bar_width/2]
    for x in x1:
        for i in range(2):
            ax1.text(x + xs[i], perceived_fast_pos[i][x], f'{perceived_fast[i][x]:.2f}', ha='center', va='bottom', fontsize=11)
            ax1.text(x + xs[i], perceived_fast_pos[i][x] + evict_fast_pos[i][x], f'{perceived_fast[i][x] + evict_fast[i][x]:.2f}', ha='center', va='bottom', fontsize=11)

    ax1.set_ylabel('fast - Latency (s)')
    ax1.set_xticks(x1)
    # ax1.set_xticklabels(categories1)
    ax1.set_xticklabels([])
    ax1.set_ylim(0, 0.14)
    ax1.set_yticks([0, 0.04, 0.08, 0.12 ])
    # ax1.legend()

    # Second subplot
    ax2.bar(x1 - bar_width/2, perceived_fast_sh[2:4], bar_width, yerr=perceived_err_fast_sh[2:4], error_kw=error_style, color=color1)
    ax2.bar(x1 - bar_width/2, evict_fast_sh[2:4], bar_width, yerr=full_err_fast_sh[2:4], error_kw=error_style, bottom=perceived_fast_sh[2:4], color=color1, alpha=0.5)

    ax2.bar(x1 + bar_width/2, perceived_fast_mal[2:4], bar_width, yerr=perceived_err_fast_mal[2:4], error_kw=error_style, color=color2)
    ax2.bar(x1 + bar_width/2, evict_fast_mal[2:4], bar_width, yerr=full_err_fast_mal[2:4], error_kw=error_style, bottom=perceived_fast_mal[2:4], color=color2, alpha=0.5)

    for x in x1:
        for i in range(2):
            ax2.text(x + xs[i], perceived_fast_pos[i][x + 2], f'{perceived_fast[i][x + 2]:.2f}', ha='center', va='bottom', fontsize=11)
            ax2.text(x + xs[i], perceived_fast_pos[i][x + 2] + evict_fast_pos[i][x + 2], f'{perceived_fast[i][x + 2] + evict_fast[i][x + 2]:.2f}', ha='center', va='bottom', fontsize=11)


    ax2.set_xticks(x2)
    # ax2.set_xticklabels(categories2)
    ax2.set_xticklabels([])
    ax2.set_ylim(0, 1.4)
    ax2.set_yticks([0, 0.4, 0.8, 1.2])
    # ax2.legend()

    ax3.bar(x1 - bar_width/2, perceived_slow_sh[0:2], bar_width, yerr=perceived_err_slow_sh[0:2], error_kw=error_style, color=color1)
    ax3.bar(x1 - bar_width/2, evict_slow_sh[0:2], bar_width, yerr=full_err_slow_sh[0:2], error_kw=error_style, bottom=perceived_slow_sh[0:2], color=color1, alpha=0.5)

    ax3.bar(x1 + bar_width/2, perceived_slow_mal[0:2], bar_width, yerr=perceived_err_slow_mal[0:2], error_kw=error_style, color=color2)
    ax3.bar(x1 + bar_width/2, evict_slow_mal[0:2], bar_width, yerr=full_err_slow_mal[0:2], error_kw=error_style, bottom=perceived_slow_mal[0:2], color=color2, alpha=0.5)

    perceived_slow = [perceived_slow_sh, perceived_slow_mal]
    evict_slow = [evict_slow_sh, evict_slow_mal]
    perceived_slow_pos = [list_add(perceived_slow_sh, perceived_err_slow_sh), list_add(perceived_slow_mal, perceived_err_slow_mal)]
    evict_slow_pos = [list_add(evict_slow_sh, full_err_slow_sh), list_add(evict_slow_mal, full_err_slow_mal)]
    for x in x1:
        for i in range(2):
            ax3.text(x + xs[i], perceived_slow_pos[i][x], f'{perceived_slow[i][x]:.2f}', ha='center', va='bottom', fontsize=11)
            ax3.text(x + xs[i], perceived_slow_pos[i][x] + evict_slow_pos[i][x], f'{perceived_slow[i][x] + evict_slow[i][x]:.2f}', ha='center', va='bottom', fontsize=11)



    ax3.set_xticks(x1)
    # ax3.set_xlabel('Fast Network')
    ax3.set_ylabel('slow - Latency (s)', labelpad = 10)
    ax3.set_xticklabels(categories1)
    ax3.set_ylim(0, 1.6)
    ax3.set_yticks([0, 0.4, 0.8, 1.2])
    #

    ax4.bar(x1 - bar_width/2, perceived_slow_sh[2:4], bar_width, yerr=perceived_err_slow_sh[2:4], error_kw=error_style, color=color1)
    ax4.bar(x1 - bar_width/2, evict_slow_sh[2:4], bar_width, yerr=full_err_slow_sh[2:4], error_kw=error_style, bottom=perceived_slow_sh[2:4], color=color1, alpha=0.5)

    ax4.bar(x1 + bar_width/2, perceived_slow_mal[2:4], bar_width, yerr=perceived_err_slow_mal[2:4], error_kw=error_style, color=color2)
    ax4.bar(x1 + bar_width/2, evict_slow_mal[2:4], bar_width, yerr=full_err_slow_mal[2:4], error_kw=error_style, bottom=perceived_slow_mal[2:4], color=color2, alpha=0.5)

    for x in x1:
        for i in range(2):
            ax4.text(x + xs[i], perceived_slow_pos[i][x + 2], f'{perceived_slow[i][x + 2]:.2f}', ha='center', va='bottom', fontsize=11)
            ax4.text(x + xs[i], perceived_slow_pos[i][x + 2] + evict_slow_pos[i][x + 2], f'{perceived_slow[i][x + 2] + evict_slow[i][x + 2]:.2f}', ha='center', va='bottom', fontsize=11)


    # ax4.set_xlabel('Slow Network')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(categories2)
    ax4.set_ylim(0, 9)
    ax4.set_yticks([0, 2, 4, 6, 8])

    left  = 0.12  # the left side of the subplots of the figure
    right = 0.99   # the right side of the subplots of the figure
    bottom = 0.08  # the bottom of the subplots of the figure
    top = 0.85   # the top of the subplots of the figure
    wspace = 0.20  # the amount of width reserved for blank space between subplots
    hspace = 0.10   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig.legend(loc='upper center', bbox_to_anchor=(0.56, 1.03), ncol=2, frameon=False)

    # Display the plot
    os.makedirs('./eval_fig', exist_ok=True)
    plt.savefig('./eval_fig/figure7.pdf') 

def render_latency_breakdown():
    datasets = ["laion", "sift", "trip", "msmarco", "laion_mal", "sift_mal", "trip_mal", "msmarco_mal"]
    network = ["fast", "slow"]

    raw_data = []
    # laion
    for d in datasets:
        raw_data_of_d = []
        for n in network:
            f_name = "./script/artifact/results/latency_" + n + "_" + d + ".fvecs"
            l = fvecs_read(f_name)[0]
            nq = int(l.shape[0] / 2)
            per_l = l[0:nq]
            full_l = l[nq:]
            raw_data_of_d.append(per_l)
            raw_data_of_d.append(full_l)
            # exit()
        raw_data.append(raw_data_of_d)

    draw_latency_breakdown(raw_data)

def render_figure7():

    print("Rendering figure 7...")
    os.chdir(os.path.expanduser('~/compass/'))

    datasets = ["laion", "sift", "trip", "msmarco", "laion_mal", "sift_mal", "trip_mal", "msmarco_mal"]
    network = ["fast", "slow"]
    raw_data = []
    # laion
    for d in datasets:
        raw_data_of_d = []
        for n in network:
            f_name = "./script/artifact/results/latency_" + n + "_" + d + ".fvecs"
            l = fvecs_read(f_name)[0]
            nq = int(l.shape[0] / 2)
            per_l = l[0:nq]
            full_l = l[nq:]
            raw_data_of_d.append(per_l)
            raw_data_of_d.append(full_l)
            # print(d, n)
            # print("perceived latency: ", per_l.mean())
            # print("full latency: ", full_l.mean())
            # print("perceived latency error: ", np.std(per_l))
            # print("full latency error: ", np.std(full_l))
            # exit()
        raw_data.append(raw_data_of_d)

    draw_latency_breakdown(raw_data)
    print(" -> Done.")


if __name__ == "__main__":
    render_figure7()
    
    

    

