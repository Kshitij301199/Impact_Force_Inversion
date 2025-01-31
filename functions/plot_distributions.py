import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({
    'font.size': 7,             # Set global font size
    'font.family': 'Arial',      # Set global font family
    'legend.fontsize': 8,        # Set legend font size
    'figure.figsize': (8, 5) # Set figure size in inches
})

def main(model_type, interval, task, config):
    output_dir = f"../{task}/dist_plots/{config}" 
    os.makedirs(output_dir, exist_ok= True)
    data = pd.read_csv(f"../{task}/model_evaluation/best_combinations.csv", index_col=False)
    df = data[(data['Interval'] == interval) & (data['Model'] == model_type)]
    df.reset_index(drop=True, inplace= True)

    preds, trues = [], []
    for i, row in df.iterrows():
        t, v = int(row['Test']), int(row['Val'])
        temp = pd.read_csv(f"../{task}/output_df/{config}/{interval}/{model_type}_t{t}_v{v}.csv")
        trues.extend(temp['Output'].to_numpy())
        preds.extend(temp['Predicted_Output'].to_numpy())

    bins = np.arange(10, 350, 5)

    plt.hist(trues, bins= bins, color= "blue", alpha= 0.8, label='True');
    plt.hist(preds, bins= bins, color= "red", alpha= 0.8, label='Predicted');
    plt.xlabel("Normal Force [kN]")
    plt.ylabel("Frequency")
    plt.legend(loc='best')
    plt.savefig(f"{output_dir}/{interval}_{model_type}_{config}_dist.png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help= "interval seconds")
    parser.add_argument("--model_type", type=str, default='LSTM', help= "model type")
    parser.add_argument("--task", type=str, default="comparison_baseline", help="input the name of task")
    parser.add_argument("--config", type=str, default="default", help="input name of the config")

    args = parser.parse_args()
    print(f"Running main with {args.interval} {args.model_type} {args.task}")
    main(args.model_type, args.interval, args.task)