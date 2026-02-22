import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data
df = pd.read_csv('medical_examination.csv')

# 2. Add overweight column (BMI > 25)
height_m = df['height'] / 100
bmi = df['weight'] / (height_m ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalize data (0 = good, 1 = bad)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # 4. Create DataFrame for categorical plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 5. Group and reformat data
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # 6. Draw the categorical plot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # 7. Return the figure
    return fig


def draw_heat_map():
    # 8. Clean the data
    h_low, h_high = df['height'].quantile([0.025, 0.975])
    w_low, w_high = df['weight'].quantile([0.025, 0.975])

    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'].between(h_low, h_high)) &
        (df['weight'].between(w_low, w_high))
    ]

    # 9. Calculate correlation matrix
    corr = df_heat.corr()

    # 10. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 11. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 12. Plot the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 13. Return the figure
    return fig
