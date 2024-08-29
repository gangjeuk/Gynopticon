from typing import *
import pandas as pd
import numpy as np
#import altair as alt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from simul import simulate_with_liar, simulate_without_liar
from utils import Config
import seaborn as sns


def boxplot(fig = None, ax = None, do_lie = False, tactic: Literal["random", "select"] = "random"):
    if do_lie is True:
        benign, cheater, _ = simulate_with_liar(model_acc=0.8, played_match=20, vote_per_match=1, benign_num=2, cheater_num=1, tactic=tactic)
    elif do_lie is False:
        benign, cheater, _ = simulate_without_liar(model_acc=0.8, played_match=20, vote_per_match=1, benign_num=2, cheater_num=1)

    benign_dub = [benign[key][0] for key in benign.keys()]
    cheater_dub = [cheater[key][0] for key in cheater.keys()]
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.boxplot([benign_dub, cheater_dub])
    ax.set_ylim(-2.5, 2.5)
    ax.set_ylabel("Dubious", labelpad=0.0)
    ax.set_xticks([1, 2], ["Benign user", "Cheating User"])
    return fig, ax 


def scatter_and_line(fig = None, ax = None,  do_lie = False, tactic: Literal["random", "select"] = "random"):
    model_acc = np.linspace(1.0, 0.5, 100).tolist()

    benign_scat, cheater_scat = {"acc": [], "dub": []}, {"acc": [], "dub": []}
    benign_line, cheater_line = [], []

    for acc in model_acc:
        # function call below assume
        # played 10 games and 2 votes per each game
        if do_lie is True:
            benign, cheater, _ = simulate_with_liar(model_acc=acc, played_match=10, vote_per_match=1, benign_num=2, cheater_num=1, tactic=tactic)
        elif do_lie is False:
            benign, cheater, _ = simulate_without_liar(model_acc=acc, played_match=10, vote_per_match=1, benign_num=2, cheater_num=1)
        benign_dub = [benign[key][0] for key in benign.keys()]
        cheater_dub = [cheater[key][0] for key in cheater.keys()]
        # scatter
        benign_scat["acc"] += [acc for _ in benign_dub]
        benign_scat["dub"] += benign_dub
        cheater_scat["acc"] += [acc for _ in cheater_dub]
        cheater_scat["dub"] += cheater_dub
        # line
        benign_line.append(sum(benign_dub) / len(benign_dub))
        cheater_line.append(sum(cheater_dub) / len(cheater_dub))

    if fig is None and ax is None:  
        fig, ax = plt.subplots()
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylabel("Dubious", labelpad=0.0)
    ax.set_xlabel("Model Acc")

    # plot
    ax.plot(model_acc, benign_line, label="Benign user")
    ax.plot(model_acc, cheater_line, label="Cheating user")
    ax.legend(loc='upper left')

    # scatter
    # ax.scatter('acc', 'dub', data=benign_scat)
    # ax.scatter('acc', 'dub', data=cheater_scat)

    # trendline
    z = np.polyfit(benign_scat["acc"], benign_scat["dub"], 1)
    p = np.poly1d(z)
    ax.plot(benign_scat["acc"], p(benign_scat["acc"]), "b--")
    z = np.polyfit(cheater_scat["acc"], cheater_scat["dub"], 1)
    p = np.poly1d(z)
    ax.plot(cheater_scat["dub"], p(cheater_scat["dub"]), "r--")

    return ax


def contour(fig = None, ax = None, do_lie = False, tactic: Literal["random", "select"] = "random"):
    TOTAL_USER = 100
    vote_cnt = 2
    match_cnt = 1
    model_acc = np.linspace(1.0, 0.5, TOTAL_USER // 2).tolist()
    cheat_rate = []
    sim_acc = []
    
    for cheat_num in range(TOTAL_USER // 2):
        cheat_rate.append(cheat_num / TOTAL_USER)
        sim_ret = []
        for acc in model_acc:
            if do_lie is True:
                benign, cheater, _ = simulate_with_liar(model_acc=acc, played_match=match_cnt, vote_per_match=vote_cnt, benign_num=TOTAL_USER - cheat_num, cheater_num=cheat_num, tactic=tactic)
            elif do_lie is False:
                benign, cheater, _ = simulate_without_liar(model_acc=acc, played_match=match_cnt, vote_per_match=vote_cnt, benign_num=TOTAL_USER - cheat_num, cheater_num=cheat_num)
            
            benign_dub = [benign[key][0] for key in benign.keys()]
            cheater_dub = [cheater[key][0] for key in cheater.keys()]
            correct = len(list(filter(lambda x: x < 0, benign_dub))) + len(
                list(filter(lambda x: x > 0, cheater_dub))
            )
            sim_ret.append(correct / (TOTAL_USER * match_cnt))
        sim_acc.append(sim_ret)

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel("Cheater Rate")
    ax.set_xlabel("Model Acc")
    X, Y = np.meshgrid(model_acc, cheat_rate)
    co = ax.contourf(X, Y, sim_acc, levels=np.linspace(0.5, 1.0, 11), extend='min')
    fig.colorbar(co, ax=ax)
    
    return fig, ax


def plot_one_third_cheater(fig = None, ax = None, do_lie = False, tactic: Literal["random", "select"] = "random"):

    TOTAL_USER = 100
    cheat_user = 33
    vote_cnt = 2
    match_cnt = 10
    model_acc = np.linspace(1.0, 0.5, TOTAL_USER // 2).tolist()

    sim_ret = []
    for acc in model_acc:
        if do_lie is True:
            benign, cheater, _ = simulate_with_liar(model_acc=acc, played_match=match_cnt, vote_per_match=vote_cnt, benign_num=TOTAL_USER - cheat_user, cheater_num=cheat_user, tactic=tactic)
        elif do_lie is False:
            benign, cheater, _ = simulate_without_liar(model_acc=acc, played_match=match_cnt, vote_per_match=vote_cnt, benign_num=TOTAL_USER - cheat_user, cheater_num=cheat_user)
        
        benign_dub = [benign[key][0] for key in benign.keys()]
        cheater_dub = [cheater[key][0] for key in cheater.keys()]
        correct = len(list(filter(lambda x: x < 0, benign_dub))) + len(
            list(filter(lambda x: x > 0, cheater_dub))
        )
        sim_ret.append(correct / (TOTAL_USER * match_cnt))

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    ax.plot(model_acc, sim_ret, 'o-')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model Acc")

    return fig, ax


def figure_1():
    global LIE_TYPE, LIE_FREQ
    LIE_TYPE = 'random'
    
    fig, axs = plt.subplots(2, 3)
    boxplot(fig, axs[0, 0])
    boxplot(fig, axs[0, 1], do_lie=True, tactic="random")
    boxplot(fig, axs[0, 2], do_lie=True, tactic="select")
    
    scatter_and_line(fig, axs[1, 0])
    scatter_and_line(fig, axs[1, 1], do_lie=True, tactic="random")
    scatter_and_line(fig, axs[1, 2], do_lie=True, tactic="select")

    axs[0, 0].set_title('(1) Without liar', fontdict={'fontsize': 'x-large'})
    axs[0, 1].set_title('(2) With random liar', fontdict={'fontsize': 'x-large'})
    axs[0, 2].set_title('(3) With tactical liar', fontdict={'fontsize': 'x-large'})
    axs[1, 0].set_title('(1) Without liar', fontdict={'fontsize': 'x-large'})
    axs[1, 1].set_title('(2) With random liar', fontdict={'fontsize': 'x-large'})
    axs[1, 2].set_title('(3) With tactical liar', fontdict={'fontsize': 'x-large'})
    
    plt.suptitle("(a) Dubious score after simulation with fixed model acc (80%)", fontsize='xx-large', fontweight='bold')
    # Adjust vertical_spacing = 0.5 * axes_height
    plt.subplots_adjust(hspace=0.5)

    # Add text in figure coordinates
    plt.figtext(0.5, 0.485, '(b) Distribution of dubious score', ha='center', va='center', fontdict={'fontsize': 'xx-large', 'fontweight': 'bold'})
    fig.set_figwidth(13)
    fig.set_figheight(7)
    plt.savefig(fname='img/figure1.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(fname='img/figure1.png', bbox_inches='tight', pad_inches=0)
    
def figure_2():

    fig, axs = plt.subplots(2, 3)
    plot_one_third_cheater(fig, axs[0, 0])
    plot_one_third_cheater(fig, axs[0, 1], do_lie=True, tactic="random")
    plot_one_third_cheater(fig, axs[0, 2], do_lie=True, tactic="select")
    
    contour(fig, axs[1, 0])
    contour(fig, axs[1, 1], do_lie=True, tactic="random")
    contour(fig, axs[1, 2], do_lie=True, tactic="select")

    axs[0, 0].set_title('(1) Without liar', fontdict={'fontsize': 'x-large'})
    axs[0, 1].set_title('(2) With random liar' , fontdict={'fontsize': 'x-large'})
    axs[0, 2].set_title('(3) With tactical liar' , fontdict={'fontsize': 'x-large'})
    axs[1, 0].set_title('(1) Without liar' , fontdict={'fontsize': 'x-large'})
    axs[1, 1].set_title('(2) With random liar' , fontdict={'fontsize': 'x-large'})
    axs[1, 2].set_title('(3) With tactical liar' , fontdict={'fontsize': 'x-large'})
    
    plt.suptitle("(a) Accuracy with fixed cheater rate (33%)", fontsize='xx-large', fontweight='bold')
    # Adjust vertical_spacing = 0.5 * axes_height
    plt.subplots_adjust(hspace=0.5)

    # Add text in figure coordinates
    plt.figtext(0.5, 0.485, '(b) Contour of accuracy', ha='center', va='center', fontdict={'fontsize': 'xx-large', 'fontweight': 'bold'})
    
    fig.set_figwidth(13)
    fig.set_figheight(7)
    plt.savefig(fname='img/figure2.png', bbox_inches='tight', pad_inches=0)  
    plt.savefig(fname='img/figure2.pdf', bbox_inches='tight', pad_inches=0)  
    
'''
def figure_appendix(playdata, e, cheater):
    for bat in playdata['battle']:
        bat['parti'] = str(bat['parti'])
        
    source = pd.DataFrame([bat for bat in playdata['battle']])
    chart = alt.Chart(source).mark_bar().encode(
        x='start',
        x2='end',
        y='parti'
    )
    
    chart.save(f'img/battle/{playdata["game"]}-battle.png')
    
    if len(playdata['votes']) == 0:
        return
    source = pd.DataFrame([vote for vote in playdata['votes']])
    chart = alt.Chart(source).mark_bar().encode(
        x='start',
        x2='end',
        y='target'
    )
    
    chart.save(f'img/battle/{playdata["game"]}-vote.png')
    return
'''
def boxplot_figure_3(fig = None, ax = None, normal_score = (), cheater_score = ()):
    
    if ax is None:
        fig, ax = plt.subplots()
    

    ax.boxplot([normal_score[0], cheater_score[0]], labels=["Benign user", "Cheating User"])
    ax.set_ylabel("Dubious score", labelpad=0.0)
    
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=16, label='Zoomed', alpha=0.4)
    
    FN_cases = np.where(normal_score[0] > 0, True, False)
    TP_cases = np.where(normal_score[0] < 0, True, False)
    FN = np.where(FN_cases, normal_score[0], None)
    TP = np.where(TP_cases, normal_score[0], None)
    ax.scatter([1 for _ in range(len(TP[TP != None]))], TP[TP != None],alpha=0.4, color='g', label="True Positive/Negative")
    ax.scatter([1 for _ in range(len(FN[FN != None]))], FN[FN != None], alpha=0.4, color='r', label="False Negative")

    FP_cases = np.where(cheater_score[0] < 0, True, False)
    TN_cases = np.where(cheater_score[0] > 0, True, False)
    FP = np.where(FP_cases, cheater_score[0], None)
    TN = np.where(TN_cases, cheater_score[0], None)
    ax.scatter([2 for _ in range(len(FP[FP != None]))], FP[FP != None], alpha=0.4, color='y', label="False Positive")
    ax.scatter([2 for _ in range(len(TN[TN != None]))], TN[TN != None],alpha=0.4, color='g')

    ax.axhline(y=0, color='#ff3300', linestyle='--', linewidth=1, label='Threshold')
    ax.set_ylim(-0.5, 4.5)
    
    # Zooming
    axins = zoomed_inset_axes(ax, 4, loc='upper left', axes_kwargs={'facecolor': 'lightgray'})
    # for labeling
    axins.axhline(y=0, color='lightgray', linestyle='-', linewidth=7, label='Zoomed', alpha=0.4)
    axins.axhline(y=0, color='#ff3300', linestyle='--', linewidth=1, label='Threshold')
    
    axins.scatter([0.05 for _ in range(len(TP[TP != None]))], TP[TP != None],alpha=0.8, color='g', vmin=0.2, vmax=0.2, label="True Positive/Negative")
    axins.scatter([0.05 for _ in range(len(FN[FN != None]))], FN[FN != None], alpha=0.8, color='r', vmin=0.2, vmax=0.2, label="False Negative")
    axins.scatter([0.15 for _ in range(len(FP[FP != None]))], FP[FP != None], alpha=0.8, color='y', vmin=0.2, vmax=0.2, label="False Positive")
    axins.scatter([0.15 for _ in range(len(TN[TN != None]))], TN[TN != None],alpha=0.8, color='g', vmin=0.2, vmax=0.2)
    

    
    axins.set_ylim(-0.2, 0.2)
    axins.set_xlim(0, 0.2)
    # ticks invisible
    axins.set_xticks([])
    axins.set_yticks([])
    axins.grid()
    
    
    return fig, ax 

def figure_3():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    from eval.eval import res as eval_res
    from eval.lie import res as lie_res

    fig = plt.figure()
    gs = fig.add_gridspec(3,3)
    sns.set_palette('bright')

    # Confusion matrix
    from matplotlib.colors import ListedColormap
    # Red, Orange, Green
    cmap = ListedColormap([[0.973,0.796,0.678],[1,0.91,0.592],[0.765,0.882,0.698]] + [[0.765,0.882,0.698]]*35)
    display_labels = ["N", "Ch"]
    tn, tp, fn, fp = np.sum(eval_res[:,4:8], axis=0, dtype=int)
    ax = fig.add_subplot(gs[1, 2])
    ConfusionMatrixDisplay(confusion_matrix=np.array([[tp,fn], [fp, tn]]), display_labels=display_labels).plot(
    include_values=True, cmap=cmap, ax=ax, colorbar=False, text_kw={'fontsize': 'xx-large', 'color':'black'})
    ax.set_title("Without liar")
    ax.xaxis.set_ticklabels(['','',])
    ax.set_xlabel('')
    ax.tick_params(axis='x', which='both',bottom=False)
    
    ax = fig.add_subplot(gs[2, 2])
    tn, tp, fn, fp = np.sum(lie_res[:,4:8], axis=0, dtype=int)
    ConfusionMatrixDisplay(confusion_matrix=np.array([[tp,fn], [fp, tn]]), display_labels=display_labels).plot(
    include_values=True, cmap=cmap, ax=ax, colorbar=False, text_kw={'fontsize': 'xx-large', 'color':'black'})
    ax.set_title("With liar")

    n_dubs, c_dubs = np.array([]), np.array([])
    n_vals, c_vals = np.array([]), np.array([])
    # Boxplot
    # result of eval/eval.py
    for i, r in enumerate(eval_res):
        s_agg, t_agg, _, thresh_acc = r[8:]
        
        normal_dub = np.where(np.invert(t_agg), s_agg, None)
        normal_dub = normal_dub[normal_dub != None] - thresh_acc

        n_dubs = np.append(n_dubs, normal_dub)
        
        cheater_score = np.where(t_agg, s_agg, None)
        cheater_score = cheater_score[cheater_score != None] - thresh_acc

        
        c_dubs = np.append(c_dubs, cheater_score)
        
    ax = fig.add_subplot(gs[:, 0])
    boxplot_figure_3(fig, ax, (n_dubs, n_vals), (c_dubs, c_vals))
    ax.set_title("Without liar", fontdict={'fontsize': 'x-large'})

    n_dubs, c_dubs = np.array([]), np.array([])
    n_vals, c_vals = np.array([]), np.array([])
    # result of eval/eval.py
    for i, r in enumerate(lie_res):
        s_agg, t_agg, _, thresh_acc = r[8:]
        
        normal_dub = np.where(np.invert(t_agg), s_agg, None)
        normal_dub = normal_dub[normal_dub != None] - thresh_acc

        n_dubs = np.append(n_dubs, normal_dub)
        
        cheater_score = np.where(t_agg, s_agg, None)
        cheater_score = cheater_score[cheater_score != None] - thresh_acc

        c_dubs = np.append(c_dubs, cheater_score)
    ax = fig.add_subplot(gs[:, 1])
    boxplot_figure_3(fig, ax, (n_dubs, n_vals), (c_dubs, c_vals))
    ax.set_title("With liar", fontdict={'fontsize': 'x-large'})
    # Title and legend
    plt.legend(bbox_to_anchor=(4.9, .8), loc='right', borderaxespad=0., framealpha=1, facecolor ='white', frameon=True)
    
    def set_title(rect_left = (0.13, -0.05, 0.5, 0.0), rect_right = (0.68, -0.05, 0.2, 0.0)):
        #rect_left = 0, 0, 0.5, 0.8  # x, y, width, height
        #rect_right = 0.5, 0, 0.5, 0.8
        ax_left = fig.add_axes(rect_left)
        ax_right = fig.add_axes(rect_right)
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.spines['right'].set_visible(False)
        ax_left.spines['top'].set_visible(False)
        ax_left.spines['bottom'].set_visible(False)
        ax_left.spines['left'].set_visible(False)
        ax_left.set_axis_off()
        ax_right.set_xticks([])
        ax_right.set_yticks([])
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['bottom'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.set_axis_off()
        ax_left.set_title('(a) Standardized dubious scores', fontdict={'fontsize': 'xx-large', 'fontweight': 'bold'})
        ax_right.set_title('(b) Results', fontdict={'fontsize': 'xx-large', 'fontweight': 'bold'})
    set_title()
    
    #axs[0].set_title("Without liar", fontdict={'fontsize': 'x-large'})
    #axs[1].set_title("With liar", fontdict={'fontsize': 'x-large'})

    fig.set_figwidth(9.6)
    fig.set_figheight(6)
    fig.savefig('img/figure3.pdf', bbox_inches='tight', pad_inches=0.1)
    fig.savefig('img/figure3.png', bbox_inches='tight', pad_inches=0.1)
    fig.savefig('img/figure3.eps', bbox_inches='tight', pad_inches=0.1)
    fig.savefig('img/figure3.svg', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()