import plotly.express as px
import numpy as np


PALETTE = [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
    "#B3B3B3", "#E5C494", "#9C9DBB", "#E6946B", "#DA8DC4", "#AFCC63",
    "#F2D834", "#E8C785", "#BAB5AE", "#D19C75", "#AC9AAD", "#CD90C5",
    "#B8C173", "#E5D839", "#ECCA77", "#C1B7AA", "#BBA37E", "#BC979D",
    "#C093C6", "#C1B683", "#D8D83E", "#F0CD68", "#C8BAA5", "#A6AB88",
    "#CC958F", "#B396C7", "#CBAB93", "#CCD844", "#F3D05A", "#CFBCA1",
    "#90B291", "#DC9280", "#A699C8", "#D4A0A3", "#BFD849", "#F7D34B",
    "#D6BF9C", "#7BBA9B", "#EC8F71", "#999CC9", "#DD95B3", "#B2D84E",
    "#FBD63D", "#DDC198"
]


def get_col_i(pal, i):
    return pal[i % len(pal)]


def scatterplot(x, y, labels, title=None, return_fig=False):
    """Beautiful scatter plots.

    Parameters
    __________
    x, y: ndarray
        x and y coordinates
    labels: ndarray
        Cluster IDs or annotations.
    """
    unq_labels = np.unique(labels)
    max_label = unq_labels.size

    fig = px.scatter(
        x=x,
        y=y,
        category_orders={'color': unq_labels.astype(str)},
        opacity=1,
        color_discrete_map={str(i): get_col_i(PALETTE, i)
                            for i in range(max_label)},
        color=labels.astype(str),
    )

    fig.update_layout(
        showlegend=True,
        plot_bgcolor='#FFFFFF',
        xaxis={'visible': False},
        yaxis={'visible': False},
        title=title,
    )
    
    if return_fig:
        return fig
    fig.show()


def heatmap(hm, classes=None, exclude_diag=True, log1p=True):
    """Plot a heatmap.

    Parameters
    __________
    hm: ndarray
        The matrix to plot.
    classes: None or ndarray
        Contains labels for the axis.
    exclude_diag: bool
        If True, will make sure the diagonal is not visible.
    log1p: bool
        Whether to log-transform the data for better visibility.
    """
    if log1p:
        hm = np.log(hm + 1)

    width = height = 600
    if classes is not None:
        n_classes = len(classes)
        width = height = max(n_classes * 10, 800)
        font_size = max(3, 13 - n_classes // 10)
    zmin = None
    if exclude_diag:
        zmin = np.min(hm[~np.eye(hm.shape[0], dtype=bool)])
    fig = px.imshow(hm, x=classes, y=classes,
                    width=width, height=height, zmin=zmin)
    if classes is not None:
        fig.update_xaxes(tickfont_size=font_size)
        fig.update_yaxes(tickfont_size=font_size)

    fig.show()
