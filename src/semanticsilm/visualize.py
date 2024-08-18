import os
from pyvis.network import Network
from llama_index.core import load_index_from_storage
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from community import community_louvain


def visualize_networkx(g, output_file='silmarillion_graph_networkx.png'):
    plt.figure(figsize=(20,20))
    pos = nx.spring_layout(g, k=0.5, iterations=50)
    nx.draw(g, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=8, font_weight='bold', 
            edge_color='gray', width=1, alpha=0.7)

    edge_labels = nx.get_edge_attributes(g, 'relationship')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Silmarillion Knowledge Graph", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"NetworkX graph saved to {output_file}")

def visualize_plotly(g, output_file='silmarillion_graph_plotly.html'):
    pos = nx.spring_layout(g)

    edge_x = []
    edge_y = []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[node for node in g.nodes()],
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Silmarillion Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Interactive Silmarillion Knowledge Graph",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.write_html(output_file)
    print(f"Plotly graph saved to {output_file}")

def create_interactive_graph(G, output_file='interactive_silmarillion_graph.html'):
    # Convert to undirected graph if it's directed
    if isinstance(G, nx.DiGraph):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    partition = community_louvain.best_partition(G_undirected)
    
    node_x, node_y = zip(*pos.values())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Community',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"{node}<br># of connections: {len(adjacencies)}")
    
    node_trace.marker.color = list(partition.values())
    node_trace.hovertext = node_text

    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(edge[2].get('relationship', ''))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Silmarillion Knowledge Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Add buttons for interactivity
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{"visible": [True, True]}],
                         label="Show All",
                         method="update"),
                    dict(args=[{"visible": [True, "legendonly"]}],
                         label="Hide Nodes",
                         method="update"),
                    dict(args=[{"visible": ["legendonly", True]}],
                         label="Hide Edges",
                         method="update")
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    # Save to HTML file
    fig.write_html(output_file)
    print(f"Interactive graph saved to {output_file}")

def main():
    storage_context = StorageContext.from_defaults(
        graph_store=SimpleGraphStore.from_persist_dir(f"../../index/08_18_2024_10_32"),
        index_store=SimpleIndexStore.from_persist_dir(f"../../index/08_18_2024_10_32")
    )
    graph_store = storage_context.graph_store
    
    g = nx.DiGraph()
    for subject, relations in graph_store._data.graph_dict.items():
        for relation, object in relations:
            g.add_edge(subject, object, relationship=relation)
    
    visualize_networkx(g)
    visualize_plotly(g)
    create_interactive_graph(g)

if __name__ == "__main__":
    main()
