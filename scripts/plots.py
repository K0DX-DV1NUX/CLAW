import sys
sys.path.append(".")

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns

model = torch.load("checkpoints/ETTh1_512_192_CLAW_ETTh1_ftS_sl512_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtlinear_True_Exp/checkpoint.pth")

dw1 = model['swt.0.conv_DW.weight']
dw2 = model['swt.1.conv_DW.weight']
dw3 = model['swt.2.conv_DW.weight']
dw4 = model['swt.3.conv_DW.weight']

approx_dw1 = dw1[0].detach().numpy().squeeze()
approx_dw2 = dw2[0].detach().numpy().squeeze()
approx_dw3 = dw3[0].detach().numpy().squeeze()
approx_dw4 = dw4[0].detach().numpy().squeeze()

detail_dw1 = dw1[1].detach().numpy().squeeze()
detail_dw2 = dw2[1].detach().numpy().squeeze()
detail_dw3 = dw3[1].detach().numpy().squeeze()
detail_dw4 = dw4[1].detach().numpy().squeeze()

approx_filters = [approx_dw1, approx_dw2, approx_dw3, approx_dw4]
detail_filters = [detail_dw1, detail_dw2, detail_dw3, detail_dw4]




# Filter Shape Plot
subplot_titles = [f"WSR{i+1}" for i in range(4)]

fig = make_subplots(rows=4, cols=1,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.06,
                    #column_widths=[0.3, 0.7],
                    )

for i, (filt1, filt2) in enumerate(zip(approx_filters, detail_filters)):
    fig.add_trace(
        go.Scatter(y=filt1.flatten(), mode='lines', name=f'C1', showlegend=(i==0), line=dict(color='blue', width=3), opacity=0.7),
        row=i+1, col=1
    )
    fig.add_trace(
        go.Scatter(y=filt2.flatten(), mode='lines', name=f'C2', showlegend=(i==0), line=dict(color='red', width=3), opacity=0.7),
        row=i+1, col=1
    )
    # Mean line
    
    fig.add_trace(
        go.Scatter(y=[np.mean(filt1.flatten())]*len(filt1), x=np.arange(len(filt1)), mode='lines',
                   line=dict(color='blue', width=1, dash='dash'),
                   name='C1-mean', showlegend=(i == 0)),
        row=i+1, col=1
    )

    fig.add_trace(
        go.Scatter(y=[np.mean(filt2.flatten())]*len(filt2), x=np.arange(len(filt2)), mode='lines',
                   line=dict(color='red', width=1, dash='dash'),
                   name='C2-mean', showlegend=(i == 0)),
        row=i+1, col=1
    )

    fig.add_trace(
        go.Scatter(y=[0]*len(filt1), x=np.arange(len(filt1)), mode='lines',
                   line=dict(color='black', width=0.5), showlegend=False),
        row=i+1, col=1
    )

    fig.update_xaxes(showgrid=False, showticklabels=False, tickfont=dict(size=10), row=i+1, col=1)

    fig.add_annotation(
    x=0.1,  # x-position at the start of the x-axis
    y=0.4,  # y-position at the mean line
    text=f"{np.mean(filt1.flatten()):.3f}",  # mean value formatted
    showarrow=False,
    font=dict(color='blue', size=12),
    align='left',
    yanchor='top',
    xanchor='left',
    row=i+1, col=1 
    )
    fig.add_annotation(
    x=0.1,  # x-position at the start of the x-axis
    y=0.3,  # y-position at the mean line
    text=f"{np.mean(filt2.flatten()):.3f}",  # mean value formatted
    showarrow=False,
    font=dict(color='red', size=12),
    align='left',
    yanchor='top',
    xanchor='left',
    row=i+1, col=1 
    )

fig.update_layout(
    height=1400,
    width=600,
    showlegend=True,
    #title_text="WSR Filter Bank (Approx & Detail)",
    margin=dict(t=12, l=10, r=10, b=10),
    plot_bgcolor="white",   # inside plot area
    paper_bgcolor="white"   # entire figure background
)
fig.update_xaxes(showgrid=False, showticklabels=True, tickfont=dict(size=10), showline=True, linecolor="black", linewidth=1, row=4, col=1)
fig.update_yaxes(showgrid=False, showticklabels=True, range=[-0.4, 0.4], tickfont=dict(size=10), showline=True, linecolor="black", linewidth=1)
fig.write_image("wsr_filter_shapes.png", scale=2)



# Covariance Matrix of WSR Filters
approx_title = [f"WSR{i+1}.C1" for i in range(4)]
detail_title = [f"WSR{i+1}.C2" for i in range(4)]
titles = approx_title + detail_title

all_filters = np.concatenate([approx_filters, detail_filters])
#print("Filter Shape:", all_filters.shape)
cov_matrix = np.cov(all_filters, rowvar=True) # Covariance Matrix
#print(cov_matrix.shape)
mask = np.triu(np.ones_like(cov_matrix, dtype=bool), k=1)

plt.figure(figsize=(10, 10))
sns.heatmap(cov_matrix, mask=mask, cmap=sns.diverging_palette(280, 150, s=90, as_cmap=True), annot=True, annot_kws={"size": 16}, fmt=".2f", square=True, cbar_kws={"shrink": .8}, xticklabels=titles, yticklabels=titles, vmin=-0.07, vmax=0.07)
#plt.xticks(fontsize=12, fontweight='bold')  # bold and larger xtick labels
#plt.yticks(fontsize=12, fontweight='bold')  # bold and larger ytick labels
plt.tight_layout()
plt.savefig("wsr_covariance_matrix.png", dpi=300)


## Normalized Inner Product Matrix for Orthogonality
norm_filters = all_filters/ np.linalg.norm(all_filters, axis=1, keepdims=True)
inner_product_matrix = np.inner(norm_filters, norm_filters)

plt.figure(figsize=(10, 10))
sns.heatmap(inner_product_matrix, mask=mask, cmap='PiYG', annot=True, annot_kws={"size": 14}, fmt=".2f", square=True, cbar_kws={"shrink": .8}, xticklabels=titles, yticklabels=titles, vmin=-1.0, vmax=1.0)
plt.tight_layout()
plt.savefig("wsr_filter_inner_product_matrix.png", dpi=300)

## Symmetry Check
norm_filters = all_filters / np.linalg.norm(all_filters, axis=1, keepdims=True)
reversed_filters = np.flip(norm_filters, axis=1)
symmetry_corr = np.sum(norm_filters * reversed_filters, axis=1)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=titles,
    y=symmetry_corr,
    marker_color='purple',
    text=[f"{val:.2f}" for val in symmetry_corr],
    textposition='auto',
    hovertemplate="<b>%{x}</b><br>Symmetry Corr: %{y:.4f}<extra></extra>"
))

# Update layout
fig.update_layout(
    #title="Symmetry Check of WSR Filters",
    xaxis_title="WSR Block Filters",
    yaxis_title="Symmetry Correlation",
    #yaxis=dict(range=[-1, 1]),  # symmetry correlation is between -1 and 1
    plot_bgcolor="white",
    bargap=0.2,
    xaxis=dict(
        zeroline=True,
        zerolinecolor='black',
        showline=True,
        linecolor='black',
        gridcolor='lightgray'
    ),
    yaxis=dict(
        range=[-1, 1],
        zeroline=True,
        zerolinecolor='black',
        showline=True,
        linecolor='black',
        gridcolor='lightgray'
    ),
)

fig.write_image("wsr_filter_symmetry.png", scale=1)




