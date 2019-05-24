# InfoMax-ZSL
Public repo for paper 'Visual-Semantic Mutual Information Maximization for Generative Zero-Shot Learning'

<img src='./fig/fig_info.png' width='90%'>

## Prepare the data
Download the datasets from [http://www.mpi-inf.mpg.de/zsl-benchmark](http://www.mpi-inf.mpg.de/zsl-benchmark)

## Reproduce the results
- Change the path in `main.py` to your path of the datasets
- Run any script in folder `scripts/` to reproduce the results

## Visualization

After changing the path of saved weights of networks and log files, you can visualize the results:

- Run `visualization.py`  to plot the distribution of generated features:

<div align='center'><img src='./fig/visual_2d_cub.png' width='50%'></div>

- Run `plot_curve.py` to plot the accuracy curve (for ZSL) and H curve (for GZSL):

  <img src='./fig/curve_cub_g.png' width='49%'><img src='./fig/curve_cub_g.png' width='49%'>

