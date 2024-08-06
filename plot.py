from typing import *
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from simul import simulate_with_liar, simulate_without_liar
from utils import Config

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
    ax.set_ylim(-1.1, 1.1)
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
            benign, cheater, _ = simulate_with_liar(model_acc=acc, played_match=10, vote_per_match=2, benign_num=2, cheater_num=1, tactic=tactic)
        elif do_lie is False:
            benign, cheater, _ = simulate_without_liar(model_acc=acc, played_match=10, vote_per_match=2, benign_num=2, cheater_num=1)
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
    ax.set_ylim(-1.2, 1.2)
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

    axs[0, 0].set_title('(1) Without liar')
    axs[0, 1].set_title('(2) With random liar')
    axs[0, 2].set_title('(3) With tactical liar')
    axs[1, 0].set_title('(1) Without liar')
    axs[1, 1].set_title('(2) With random liar')
    axs[1, 2].set_title('(3) With tactical liar')
    
    plt.suptitle("(a) Dubious score after simulation with fixed model acc (90%)", fontsize='x-large', fontweight='bold')
    # Adjust vertical_spacing = 0.5 * axes_height
    plt.subplots_adjust(hspace=0.5)

    # Add text in figure coordinates
    plt.figtext(0.5, 0.485, '(b) Distribution of dubious score', ha='center', va='center', fontdict={'fontsize': 'x-large', 'fontweight': 'bold'})
    fig.set_figwidth(13)
    fig.set_figheight(7)
    plt.savefig(fname='img/figure1.png', bbox_inches='tight', pad_inches=0)
    
def figure_2():
    global LIE_TYPE, LIE_FREQ
    LIE_TYPE = 'random'

    fig, axs = plt.subplots(2, 3)
    plot_one_third_cheater(fig, axs[0, 0])
    plot_one_third_cheater(fig, axs[0, 1], do_lie=True, tactic="random")
    plot_one_third_cheater(fig, axs[0, 2], do_lie=True, tactic="select")
    
    contour(fig, axs[1, 0])
    contour(fig, axs[1, 1], do_lie=True, tactic="random")
    contour(fig, axs[1, 2], do_lie=True, tactic="select")

    axs[0, 0].set_title('(1) Without liar')
    axs[0, 1].set_title('(2) With random liar')
    axs[0, 2].set_title('(3) With tactical liar')
    axs[1, 0].set_title('(1) Without liar')
    axs[1, 1].set_title('(2) With random liar')
    axs[1, 2].set_title('(3) With tactical liar')
    
    plt.suptitle("(a) Accuracy with fixed cheater rate (33%)", fontsize='x-large', fontweight='bold')
    # Adjust vertical_spacing = 0.5 * axes_height
    plt.subplots_adjust(hspace=0.5)

    # Add text in figure coordinates
    plt.figtext(0.5, 0.485, '(b) Contour of accuracy', ha='center', va='center', fontdict={'fontsize': 'x-large', 'fontweight': 'bold'})
    
    fig.set_figwidth(13)
    fig.set_figheight(7)
    plt.savefig(fname='img/figure2.png', bbox_inches='tight', pad_inches=0)  
    

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


if __name__ == '__main__':
    figure_1()
    figure_2()