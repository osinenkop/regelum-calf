## Release tool.

A few usage examples:

Check all plugins against the published versions:
```text
$ python tools/release/release.py  action=check set=plugins 
[2021-03-30 18:21:05,768][__main__][INFO] - Build outputs : /home/omry/dev/rehydra/outputs/2021-03-30/18-21-05/build
[2021-03-30 18:21:05,768][__main__][INFO] - Checking for unpublished packages
[2021-03-30 18:21:06,258][__main__][INFO] - ❋ : rehydra-ax-sweeper : newer (local=1.1.5.dev1 > latest=1.1.0rc2)
[2021-03-30 18:21:06,746][__main__][INFO] - ❋ : rehydra-colorlog : newer (local=1.1.0.dev1 > latest=1.0.1)
[2021-03-30 18:21:07,232][__main__][INFO] - ❋ : rehydra-joblib-launcher : newer (local=1.1.5.dev1 > latest=1.1.2)
[2021-03-30 18:21:07,708][__main__][INFO] - ❋ : rehydra-nevergrad-sweeper : newer (local=1.1.5.dev1 > latest=1.1.0rc2)
[2021-03-30 18:21:08,161][__main__][INFO] - ❋ : rehydra-optuna-sweeper : newer (local=1.1.0.dev1 > latest=0.9.0rc2)
[2021-03-30 18:21:08,639][__main__][INFO] - ❋ : rehydra-ray-launcher : newer (local=1.1.0.dev1 > latest=0.1.4)
[2021-03-30 18:21:09,122][__main__][INFO] - ❋ : rehydra-rq-launcher : newer (local=1.1.0.dev1 > latest=1.0.2)
[2021-03-30 18:21:09,620][__main__][INFO] - ❋ : rehydra-submitit-launcher : newer (local=1.1.5.dev1 > latest=1.1.1)
```

Check specific packages (rehydra and configen) against the published versions
```text
$ python tools/release/release.py  action=check packages=[rehydra,configen]
[2021-03-30 18:21:25,423][__main__][INFO] - Build outputs : /home/omry/dev/rehydra/outputs/2021-03-30/18-21-25/build
[2021-03-30 18:21:25,423][__main__][INFO] - Checking for unpublished packages
[2021-03-30 18:21:26,042][__main__][INFO] - ❋ : rehydra-core : newer (local=1.1.0.dev6 > latest=1.1.0.dev5)
[2021-03-30 18:21:26,497][__main__][INFO] - ❋ : rehydra-configen : newer (local=0.9.0.dev8 > latest=0.9.0.dev7)
```

Build all plugins:
```shell
$ python tools/release/release.py  action=build set=plugins
[2021-03-30 18:21:40,426][__main__][INFO] - Build outputs : /home/omry/dev/rehydra/outputs/2021-03-30/18-21-40/build
[2021-03-30 18:21:40,426][__main__][INFO] - Building unpublished packages
[2021-03-30 18:21:41,280][__main__][INFO] - Building rehydra-ax-sweeper
[2021-03-30 18:21:47,237][__main__][INFO] - Building rehydra-colorlog
[2021-03-30 18:21:52,982][__main__][INFO] - Building rehydra-joblib-launcher
[2021-03-30 18:21:58,833][__main__][INFO] - Building rehydra-nevergrad-sweeper
[2021-03-30 18:22:04,618][__main__][INFO] - Building rehydra-optuna-sweeper
[2021-03-30 18:22:10,511][__main__][INFO] - Building rehydra-ray-launcher
[2021-03-30 18:22:16,487][__main__][INFO] - Building rehydra-rq-launcher
[2021-03-30 18:22:22,302][__main__][INFO] - Building rehydra-submitit-launcher
```

Publish all build articats (sdists and wheels):
```
$ twine upload /home/omry/dev/rehydra/outputs/2021-03-30/18-21-40/build/*
...
```
