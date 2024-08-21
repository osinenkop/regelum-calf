import srccode as rg
from goalagent import repo_root


@rg.main(
    config_path=repo_root / "presets" / "vpg_ddpg_ppo_reinforce", config_name="main"
)
def launch(cfg):
    scenario = ~cfg.scenario
    scenario.run()


if __name__ == "__main__":
    job_results = launch()
    pass
