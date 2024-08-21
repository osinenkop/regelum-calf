# Rehydra torchrun Launcher
This launcher aims to make it easier to launch a run multi GPU PyTorch training with rehydra. It works by creating an Elastic Agent (torchrun launcher class) and forking the main process after rehydra is initialized. \
You can read more on the internals of torchrun [here.](https://pytorch.org/docs/stable/elastic/run.html)

# Example usage
```bash
python my_app.py -m rehydra/launcher=torchrun rehydra.launcher.nproc_per_node=8
```
