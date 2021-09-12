from ray import tune
from ray.tune import CLIReporter
from ray.tune.progress_reporter import TuneReporterBase
from ray.tune.schedulers import ASHAScheduler
from ray.tune.trial_runner import TrialRunner

from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


def run_tune(train_func, exp_name: str, resources_map: dict, search_space: dict, metric_name: str, max_iter: int = 100, stopper=None, use_asha=True):
    if use_asha:
        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric=metric_name,
            mode='min',
            max_t=max_iter,
            grace_period=10,
            reduction_factor=2)
    else:
        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric=metric_name,
            mode='min',
            max_t=max_iter,
            grace_period=max_iter,
            reduction_factor=2)

    exp_path = ROOT_DIR.joinpath("out/tune/exp")
    resume = True if TrialRunner.checkpoint_exists(exp_path.joinpath(exp_name)) else False
    logger.info(f"Tuning {exp_name} with Resume = {resume}")

    reporter_clos = TuneReporterBase.DEFAULT_COLUMNS.copy()
    reporter_clos["dev_metric"] = "mse"
    reporter = CLIReporter(metric_columns=reporter_clos)

    analysis = tune.run(
        train_func,
        stop=stopper,
        name=exp_name,
        scheduler=asha_scheduler,
        verbose=1,
        sync_to_driver=False,
        resume=resume,
        local_dir=exp_path,
        resources_per_trial=resources_map,
        progress_reporter=reporter,
        config=search_space,
        max_failures=3)

    best_trial = analysis.get_best_trial(metric=metric_name, mode="min")
    best_config = best_trial.config
    best_loss = best_trial.metric_analysis[metric_name]["min"]
    best_config["best loss"] = best_loss
    logger.info(f"Best config: {best_config}")

    return best_loss, best_config
