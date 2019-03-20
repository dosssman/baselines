import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from math import ceil

def argsparser():
    parser = argparse.ArgumentParser('Do evaluation')
    parser.add_argument('--filename', type=str, default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = argsparser()

    basedir = "/home/z3r0/random/rl/openai_logs/openai-remi"
    alphaplot_dir =  os.path.join( basedir, "alphaploting")

    if not os.path.exists( alphaplot_dir):
        os.makedirs( alphaplot_dir, exist_ok=True)

    evalres_fullname = ""

    if args.filename is None:
        print( "Error, invalid filename")
        exit()
    else:
        evalres_fullname = os.path.join( alphaplot_dir, args.filename)
        if not os.path.isfile( evalres_fullname):
            print( "Error, filename not found")
            exit()


    with open( evalres_fullname, "rb") as evalresfile:
        evalres = pkl.load( evalresfile);

    alphas = []
    avg_returns = []
    returns = []

    percentile_25th = []
    percentile_75th = []

    percentile_10th = []
    percentile_90th = []

    for key, value in evalres.items():
        alphas.append( key)
        avg_returns.append( value["avg_ret"][0])
        returns.append( value["returns"])

        percentile_25th.append( sorted( value["returns"])[ceil(.25*len( value["returns"]))])
        percentile_75th.append( sorted( value["returns"])[ceil(.75*len( value["returns"]))])

        percentile_10th.append( sorted( value["returns"])[ceil(.1*len( value["returns"]))])
        percentile_90th.append( sorted( value["returns"])[ceil(.9*len( value["returns"]))])

    print( "Alphas: ", alphas)
    print( "Avg Rets: ", avg_returns)

    # Average returns
    fig1 = plt.figure(1)
    plt.bar( alphas, avg_returns, align="center", color="blue", label="Avg")
    plt.title("ReMi - Avg Returns ~ Alphas")
    plt.xlabel( "Alphas")
    plt.ylabel( "Scores")
    plt.legend()

    # Average Returns * 25/75 percentiles
    fig2 = plt.figure(2)
    plt.bar( alphas, percentile_75th, align="center", color="red", label="75th %tile")
    plt.bar( alphas, avg_returns, align="center", color="blue", label="Avg")
    plt.bar( alphas, percentile_25th, align="center", color="green", label="25th %tile")
    plt.title( "ReMi - Avg Returns ~ Alphas - 25th/75th %tile")
    plt.xlabel( "Alphas")
    plt.ylabel( "Scores")
    plt.legend()

    # Avg Retusn + 10/90 percentiles
    fig3 = plt.figure(3)
    plt.bar( alphas, percentile_90th, align="center", color="red", label="90th %tile" )
    plt.bar( alphas, avg_returns, align="center", color="blue", label="Avg")
    plt.bar( alphas, percentile_10th, align="center", color="green", label="10th %tile")
    plt.title( "ReMi - Avg Returns ~ Alphas - 10th/90th %tile")
    plt.xlabel( "Alphas")
    plt.ylabel( "Scores")
    plt.legend()

    # plt.boxplot( x=returns, vert=True)
    plt.show()
