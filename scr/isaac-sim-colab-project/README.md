# Isaac Sim (and Isaac Lab) on Colab

Unofficial instructions for running headless [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) on Google Colab.

This setup is for demo purposes only, using various hacks to run Isaac Sim on Colab. Serious development is not recommended, as Colab is not officially supported.

## Demo

Test the demo on Google Colab:

| Isaac Sim Version | Isaac Lab Version | Colab Link | Script |
| ----------------- | ----------------- | ----------- | ------ |
| 4.5.0 | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/j3soon/isaac-sim-colab/blob/main/notebooks/isaac-sim-4.5-colab.ipynb) | [install-isaac-sim-4.5.sh](https://raw.githubusercontent.com/j3soon/colab-python-version/refs/heads/main/scripts/install-isaac-sim-4.5.sh) |
| 4.5.0 | 2.1.0 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/j3soon/isaac-sim-colab/blob/main/notebooks/isaac-lab-2.1.0-colab.ipynb) | [install-isaac-lab-2.1.0.sh](https://raw.githubusercontent.com/j3soon/colab-python-version/refs/heads/main/scripts/install-isaac-lab-2.1.0.sh) |

## Developer Notes

Run the following to clean up notebooks before committing:

```sh
nb-clean clean --preserve-cell-outputs notebooks/isaac-sim-4.5-colab.ipynb
nb-clean clean --preserve-cell-outputs notebooks/isaac-lab-2.1.0-colab.ipynb
```

Sometimes Isaac Sim installation fails due to `pip install` issues. In this case, try deleting the runtime and creating a new one.
