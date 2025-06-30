# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:10:48 2025

@author: ibargiotas
"""
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
import re

def plot_3d_graph(m, names, participation):
    m = np.array(m)
    participation = np.array(participation)
    num_nodes = len(names)

    if m.shape != (num_nodes, num_nodes):
        raise ValueError("Shape of connection matrix does not match number of node names.")

    G = nx.from_numpy_array(m)
    name_map = {i: names[i] for i in range(num_nodes)}
    G = nx.relabel_nodes(G, name_map)
    pos = nx.spring_layout(G, dim=3, seed=42)

    node_coords = np.array([pos[name] for name in names])
    node_x, node_y, node_z = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]

    edge_traces = []
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    min_weight = min(weights) if weights else 0
    max_weight = max(weights) if weights else 1

    for u, v, data in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        weight = data['weight']
        edge_thickness = 5 + 5 * (weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 1

        edge_trace = go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(width=edge_thickness, color='gray'),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=10 + 20 * (participation / np.max(participation)) if np.max(participation) > 0 else 10,
            color=participation,
            colorscale='Viridis',
            colorbar=dict(title='Participation'),
            line=dict(width=0.5, color='black')
        ),
        text=names,
        hoverinfo='text'
    )

    # Compute bounding box to frame all nodes
    x_margin, y_margin, z_margin = 0.2, 0.2, 0.2
    xrange = [node_x.min() - x_margin, node_x.max() + x_margin]
    yrange = [node_y.min() - y_margin, node_y.max() + y_margin]
    zrange = [node_z.min() - z_margin, node_z.max() + z_margin]

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='3D Graph of Features',
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(range=xrange, 
                                        showbackground=False,
                                        showticklabels=False,
                                        title='',
                                        zeroline=False),
                            yaxis=dict(range=yrange, 
                                        showbackground=False,
                                        showticklabels=False,
                                        title='',
                                        zeroline=False),
                            zaxis=dict(range=zrange, 
                                        showbackground=False,
                                        showticklabels=False,
                                        title='',
                                        zeroline=False),
                            camera=dict(
                                eye=dict(x=1.25, y=1.25, z=1.25)  # better default angle
                            )
                        ),
                        margin=dict(l=0, r=0, b=0, t=30)
                    ))
    return fig

def extract_base_name(feature_name):
    return re.split(r'\(.*?\)|[<>]=?.*$', feature_name)[0].strip()


# Simulate some models (or load from file)
# Each model is represented as a dict with: 'performance' (float), 'features' (set), 'model' (sklearn model or string)
# Replace this with your actual models list

models = [{'performance f1': 0.905568019579849,
  'performance sens': 0.8571428571428571,
  'performance spec': 0.959792477302205,
  'performance auc': 0.5,
  'features': {'X_3>0.88', 'X_5 (3)>0.38', 'X_5<0.84'},
  'model': 'Model_0',
  'performance approx_auc': 0.908467667222531},
 {'performance f1': 0.9458017135862914,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9636835278858625,
  'performance auc': 0.06035760607745046,
  'features': {'X_1 (5)>0.31', 'X_2<-0.15', 'X_5<0.84'},
  'model': 'Model_1',
  'performance approx_auc': 0.9461274782286455},
 {'performance f1': 0.9458017135862914,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9636835278858625,
  'performance auc': 0.06035760607745046,
  'features': {'X_1 (5)>0.31', 'X_2<-0.15', 'X_5<0.84'},
  'model': 'Model_2',
  'performance approx_auc': 0.9461274782286455},
 {'performance f1': 0.9439238581170584,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.959792477302205,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_3<0.41', 'X_5<0.84'},
  'model': 'Model_3',
  'performance approx_auc': 0.9441819529368167},
 {'performance f1': 0.9439238581170584,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.959792477302205,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_3<0.41', 'X_5<0.84'},
  'model': 'Model_4',
  'performance approx_auc': 0.9441819529368167},
 {'performance f1': 0.9464259503889624,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9649805447470817,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_5',
  'performance approx_auc': 0.9467759866592551},
 {'performance f1': 0.9464259503889624,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9649805447470817,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_6',
  'performance approx_auc': 0.9467759866592551},
 {'performance f1': 0.9495343507728315,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9714656290531777,
  'performance auc': 0.06035760607745046,
  'features': {'X_2 (4)>0.16', 'X_3<0.41', 'X_5<0.84'},
  'model': 'Model_7',
  'performance approx_auc': 0.9500185288123031},
 {'performance f1': 0.9495343507728315,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9714656290531777,
  'performance auc': 0.06035760607745046,
  'features': {'X_2 (4)>0.16', 'X_3<0.41', 'X_5<0.84'},
  'model': 'Model_8',
  'performance approx_auc': 0.9500185288123031},
 {'performance f1': 0.9420382478737526,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9559014267185474,
  'performance auc': 0.46044098573281456,
  'features': {'X_1 (2)>0.71', 'X_5<0.84'},
  'model': 'Model_9',
  'performance approx_auc': 0.9422364276449879},
 {'performance f1': 0.9420382478737526,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9559014267185474,
  'performance auc': 0.06035760607745046,
  'features': {'X_1 (2)>0.71', 'X_5<0.84'},
  'model': 'Model_10',
  'performance approx_auc': 0.9422364276449879},
 {'performance f1': 0.9350572150393818,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9416342412451362,
  'performance auc': 0.5,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_11',
  'performance approx_auc': 0.9351028349082824},
 {'performance f1': 0.9464259503889624,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9649805447470817,
  'performance auc': 0.29831387808041504,
  'features': {'X_2<-0.15', 'X_5<0.84'},
  'model': 'Model_12',
  'performance approx_auc': 0.9467759866592551},
 {'performance f1': 0.9464259503889624,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9649805447470817,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_5<0.84'},
  'model': 'Model_13',
  'performance approx_auc': 0.9467759866592551},
 {'performance f1': 0.9350572150393818,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9416342412451362,
  'performance auc': 0.9396423939225496,
  'features': {'X_0 (2)<-0.38', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_14',
  'performance approx_auc': 0.9351028349082824},
 {'performance f1': 0.9350572150393818,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9416342412451362,
  'performance auc': 0.9396423939225496,
  'features': {'X_0 (2)<-0.38', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_15',
  'performance approx_auc': 0.9351028349082824},
 {'performance f1': 0.9414079795346092,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9546044098573282,
  'performance auc': 0.5,
  'features': {'X_2 (4)>0.16', 'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_16',
  'performance approx_auc': 0.9415879192143783},
 {'performance f1': 0.9414079795346092,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9546044098573282,
  'performance auc': 0.5,
  'features': {'X_2 (4)>0.16', 'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_17',
  'performance approx_auc': 0.9415879192143783},
 {'performance f1': 0.9401448347209221,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9520103761348897,
  'performance auc': 0.5,
  'features': {'X_2 (2)>0.27', 'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_18',
  'performance approx_auc': 0.9402909023531592},
 {'performance f1': 0.9388782003847861,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9494163424124513,
  'performance auc': 0.5,
  'features': {'X_0 (3)>0.33', 'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_19',
  'performance approx_auc': 0.93899388549194},
 {'performance f1': 0.9401448347209221,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9520103761348897,
  'performance auc': 0.5,
  'features': {'X_2 (2)>0.27', 'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_20',
  'performance approx_auc': 0.9402909023531592},
 {'performance f1': 0.9388782003847861,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9494163424124513,
  'performance auc': 0.5,
  'features': {'X_0 (3)>0.33', 'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_21',
  'performance approx_auc': 0.93899388549194},
 {'performance f1': 0.9388782003847861,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9494163424124513,
  'performance auc': 0.5,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_4 (1)>0.34', 'X_5<0.84'},
  'model': 'Model_22',
  'performance approx_auc': 0.93899388549194},
 {'performance f1': 0.9388782003847861,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9494163424124513,
  'performance auc': 0.5,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_4 (1)>0.34', 'X_5<0.84'},
  'model': 'Model_23',
  'performance approx_auc': 0.93899388549194},
 {'performance f1': 0.9407768424161868,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.953307392996109,
  'performance auc': 0.45330739299610895,
  'features': {'X_5 (1)<-0.68', 'X_5<0.84'},
  'model': 'Model_24',
  'performance approx_auc': 0.9409394107837687},
 {'performance f1': 0.9395119546462903,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9507133592736705,
  'performance auc': 0.7844172688530665,
  'features': {'X_5<0.84', 'X_5>1.42'},
  'model': 'Model_25',
  'performance approx_auc': 0.9396423939225496},
 {'performance f1': 0.8609052620355067,
  'performance sens': 0.7857142857142857,
  'performance spec': 0.9520103761348897,
  'performance auc': 0.5331665740226051,
  'features': {'X_5 (5)<-0.72', 'X_5<0.84'},
  'model': 'Model_26',
  'performance approx_auc': 0.8688623309245878},
 {'performance f1': 0.8609052620355067,
  'performance sens': 0.7857142857142857,
  'performance spec': 0.9520103761348897,
  'performance auc': 0.06035760607745046,
  'features': {'X_5 (5)<-0.72', 'X_5<0.84'},
  'model': 'Model_27',
  'performance approx_auc': 0.8688623309245878},
 {'performance f1': 0.9174617598061486,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9066147859922179,
  'performance auc': 0.799379284787845,
  'features': {'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_28',
  'performance approx_auc': 0.9175931072818232},
 {'performance f1': 0.9247107148224214,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.920881971465629,
  'performance auc': 0.799379284787845,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_29',
  'performance approx_auc': 0.9247267000185289},
 {'performance f1': 0.9247107148224214,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.920881971465629,
  'performance auc': 0.799379284787845,
  'features': {'X_2<-0.15', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_30',
  'performance approx_auc': 0.9247267000185289},
 {'performance f1': 0.9273190469054399,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9260700389105059,
  'performance auc': 0.9396423939225496,
  'features': {'X_2 (4)>0.16', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_31',
  'performance approx_auc': 0.9273207337409672},
 {'performance f1': 0.9273190469054399,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9260700389105059,
  'performance auc': 0.9396423939225496,
  'features': {'X_2 (4)>0.16', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_32',
  'performance approx_auc': 0.9273207337409672},
 {'performance f1': 0.9240563436763747,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9195849546044098,
  'performance auc': 0.9396423939225496,
  'features': {'X_3>0.88', 'X_4 (1)>0.34', 'X_5<0.84'},
  'model': 'Model_33',
  'performance approx_auc': 0.9240781915879193},
 {'performance f1': 0.9240563436763747,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9195849546044098,
  'performance auc': 0.9396423939225496,
  'features': {'X_3>0.88', 'X_4 (1)>0.34', 'X_5<0.84'},
  'model': 'Model_34',
  'performance approx_auc': 0.9240781915879193},
 {'performance f1': 0.9266683329167709,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9247730220492867,
  'performance auc': 0.9396423939225496,
  'features': {'X_2 (2)>0.27', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_35',
  'performance approx_auc': 0.9266722253103576},
 {'performance f1': 0.9350572150393818,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9416342412451362,
  'performance auc': 0.9396423939225496,
  'features': {'X_2 (3)>0.29', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_36',
  'performance approx_auc': 0.9351028349082824},
 {'performance f1': 0.9266683329167709,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9247730220492867,
  'performance auc': 0.9396423939225496,
  'features': {'X_2 (2)>0.27', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_37',
  'performance approx_auc': 0.9266722253103576},
 {'performance f1': 0.9350572150393818,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9416342412451362,
  'performance auc': 0.9396423939225496,
  'features': {'X_2 (3)>0.29', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_38',
  'performance approx_auc': 0.9351028349082824},
 {'performance f1': 0.9395119546462903,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9507133592736705,
  'performance auc': 0.06035760607745046,
  'features': {'X_5<0.84'},
  'model': 'Model_39',
  'performance approx_auc': 0.9396423939225496},
 {'performance f1': 0.9439238581170584,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.959792477302205,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_4 (6)>0.33', 'X_5<0.84'},
  'model': 'Model_40',
  'performance approx_auc': 0.9441819529368167},
 {'performance f1': 0.9439238581170584,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.959792477302205,
  'performance auc': 0.06035760607745046,
  'features': {'X_2<-0.15', 'X_4 (6)>0.33', 'X_5<0.84'},
  'model': 'Model_41',
  'performance approx_auc': 0.9441819529368167},
 {'performance f1': 0.9369716743289337,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9455252918287937,
  'performance auc': 0.9396423939225496,
  'features': {'X_3>0.88', 'X_5 (3)>0.38', 'X_5<0.84'},
  'model': 'Model_42',
  'performance approx_auc': 0.9370483602001112},
 {'performance f1': 0.9369716743289337,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9455252918287937,
  'performance auc': 0.9396423939225496,
  'features': {'X_3>0.88', 'X_5 (3)>0.38', 'X_5<0.84'},
  'model': 'Model_43',
  'performance approx_auc': 0.9370483602001112},
 {'performance f1': 0.9240563436763747,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9195849546044098,
  'performance auc': 0.9396423939225496,
  'features': {'X_1 (6)>0.08', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_44',
  'performance approx_auc': 0.9240781915879193},
 {'performance f1': 0.9240563436763747,
  'performance sens': 0.9285714285714286,
  'performance spec': 0.9195849546044098,
  'performance auc': 0.9396423939225496,
  'features': {'X_1 (6)>0.08', 'X_3>0.88', 'X_5<0.84'},
  'model': 'Model_45',
  'performance approx_auc': 0.9240781915879193}]

importance = {
    'X_0 (2)<-0.38': 0.1,
    'X_0 (3)>0.33': 0.12,
    'X_1 (2)>0.71': 0.15,
    'X_1 (5)>0.31': 0.2,
    'X_1 (6)>0.08': 0.22,
    'X_2<-0.15': 0.25,
    'X_2 (2)>0.27': 0.3,
    'X_2 (3)>0.29': 0.35,
    'X_2 (4)>0.16': 0.4,
    'X_3<0.41': 0.45,
    'X_3>0.88': 0.5,
    'X_4 (1)>0.34': 0.55,
    'X_4 (6)>0.33': 0.6,
    'X_5<0.84': 0.65,
    'X_5>1.42': 0.68,
    'X_5 (1)<-0.68': 0.7,
    'X_5 (3)>0.38': 0.75,
    'X_5 (5)<-0.72': 0.8
}

importance = [(feat, score) for feat, score in importance.items() if score > 0]

grouped = defaultdict(float)
for feat, score in importance:
    base = extract_base_name(feat)
    grouped[base] += score
    grouped[base] = round(grouped[base], 4) 

# Result
aggregated_importance = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

# Convert importance to a dictionary for easier access
aggregated_importance = {feat: score for feat, score in aggregated_importance}

# Get based feature names
for model in models:
    based_features = {extract_base_name(feature) for feature in model['features']}
    model['features'] = based_features

# Convert to DataFrame for easier handling
df = pd.DataFrame(models)

# Get min and max performance
min_perf, max_perf = df['performance f1'].min(), df['performance f1'].max()

# UI: Slider for performance
st.sidebar.title("Filter Models")
selected_range = st.sidebar.slider("Select performance (F1)) range:", min_value=float(min_perf),
                                   max_value=float(max_perf),
                                   value=(float(min_perf), float(max_perf)))

# Filter models by performance range
models_in_range = df[(df['performance f1'] >= selected_range[0]) & (df['performance f1'] <= selected_range[1])]

# Get all features used by filtered models
features_in_range = set().union(*models_in_range['features'])

# PRECOMPUTE feature stats based on performance-filtered models
final_models = df[
    df['performance f1'].between(selected_range[0], selected_range[1])
]

if not final_models.empty:
    all_feats = sorted(set().union(*final_models['features']))        
    feat_idx = {feat: i for i, feat in enumerate(all_feats)}
    participation = np.zeros(len(all_feats))
    m = np.zeros((len(all_feats), len(all_feats)))

    for _, row in final_models.iterrows():
        features = row['features']
        indices = [feat_idx[f] for f in features]
        for i in indices:
            participation[i] += 1
            for j in indices:
                if i != j:
                    m[i, j] += 1
else:
    all_feats, participation, m = [], [], []

# UI: Checkboxes for features with dynamic counts
st.sidebar.title("Select Features")
selected_features = []
feature_states = {}

for i, feat in enumerate(all_feats):
    count = int(participation[i])
    label = f"{feat} \n [Score : {aggregated_importance[feat]} - Count : {count}]"
    default_checked = feat in features_in_range
    checked = st.sidebar.checkbox(label=label, value=default_checked, key=feat)
    feature_states[feat] = checked
    if checked:
        selected_features.append(feat)

# ✅ NEW: keep models whose feature set is a subset of the selected ones
models_with_selected_features = df[df['features'].apply(lambda feats: set(feats).issubset(set(selected_features)))]


# Update performance range based on selected features
available_performances = models_with_selected_features['performance f1'].tolist()
if available_performances:
    new_min = min(available_performances)
    new_max = max(available_performances)
    st.write(f"Models with selected features have performance between **{new_min:.2f}** and **{new_max:.2f}**.")
else:
    st.write("No models match the selected features.")
    new_min = new_max = 0.0

# Filter again based on selected features (final models)
final_models = df[
    df['performance f1'].between(selected_range[0], selected_range[1]) &
    df['features'].apply(lambda feats: set(feats).issubset(set(selected_features)))
]

# Display final models and feature graph side by side
st.subheader("Filtered Models and Feature Graph")
col1, col2 = st.columns(2)

with col1:
    st.write(final_models[['performance f1', 'features']])

with col2:
    if not final_models.empty:
        fig = plot_3d_graph(m, all_feats, participation)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No models to display. Adjust filters or selected features.")

