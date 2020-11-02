import numpy as np
import matplotlib.pyplot as plt

#  def drawPieMarker(xs, ys, ratios, sizes, colors):
    #  #  assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'
#
    #  # calculate the points of the pie pieces
    #  for indv_ratios, x, y in zip(ratios, xs, ys):
        #  print(y)
        #  markers = []
        #  previous = 0
        #  for color, ratio in zip(colors, indv_ratios):
            #  this = 2 * np.pi * ratio + previous
            #  x  = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
            #  y  = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
            #  xy = np.column_stack([x, y])
            #  previous = this
            #  markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':color})
#
        #  # scatter each of the pie pieces to create pies
        #  print(indv_ratios)
        #  for marker in markers:
            #  ax.scatter(x, y, **marker)
#
#
#  fig, ax = plt.subplots()
#  drawPieMarker(
    #  xs=range(3),
    #  ys=range(3),
    #  ratios=[[0.3,0.3,0.3],[0.4,0.2,0.4],[.3, .2, .5]],
    #  sizes=[200,200,200],
    #  colors=['cyan', 'orange', 'teal']
#  )
#  plt.show()
#  def draw_pie(xs,ys,dist,size, colors, ax):
    #  markers = []
    #  for i in range(len(xs)):
        #  cumsum = np.cumsum(dist[i])
        #  cumsum = cumsum/ cumsum[-1]
        #  pie = [0] + cumsum.tolist()
#
        #  marker = []
        #  for r1, r2 in zip(pie[:-1], pie[1:]):
            #  angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            #  x = [0] + np.cos(angles).tolist()
            #  y = [0] + np.sin(angles).tolist()
#
            #  xy = np.column_stack([x, y])
            #  marker.append(xy)
#
        #  markers.append(np.array(marker))
#
    #  markers=np.array(markers)
#
    #  for i in range(len(colors)):
        #  ax.scatter(xs, ys, marker=markers[:,i], color=colors[i], s=size[i])
#
    #  return ax

def draw_pie(xs,ys,dist,size, colors, ax):
    for i in range(len(xs)):
        cumsum = np.cumsum(dist[i])
        cumsum = cumsum/ cumsum[-1]
        pie = [0] + cumsum.tolist()

        for r1, r2, j in zip(pie[:-1], pie[1:], range(len(dist[i]))):
            angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
            x = [0] + np.cos(angles).tolist()
            y = [0] + np.sin(angles).tolist()

            xy = np.column_stack([x, y])

            ax.plot([xs[i]], [ys[i]], marker=xy, color=colors[j], markersize=np.sqrt(size[i]))

    return ax

#  dist = [[0.3,0.3,0.3] for _ in range(10)]
dist = [
    [0.9,0.05,0.05],
    [0.3,0.3,0.4],
    [0.2,0.7,0.1],
    [0.1,0.4,0.5],
    [0.2,0.2,0.6],
    [0.9,0.05,0.05],
    [0.3,0.3,0.4],
    [0.2,0.7,0.1],
    [0.1,0.4,0.5],
    [0.2,0.2,0.6],
]
xs = range(10)
ys = range(10)
sizes = 10 * np.ones(10)
colors = ["red", "green", "blue"]

fig, ax = plt.subplots(figsize=(4,4), dpi=300)
ax.scatter([1.5], [1.5], s=10)
draw_pie(
    xs=xs,
    ys=ys,
    dist=dist,
    size=sizes,
    colors=colors,
    ax=ax,
)
plt.show()
