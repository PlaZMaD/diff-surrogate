from comet_ml import Experiment
import traceback
import sys
import os
import click
import torch
import numpy as np
sys.path.append('../')
sys.path.append('./RegressionNN')
from typing import List, Union
from model import YModel, RosenbrockModel, MultimodalSingularityModel, GaussianMixtureHumpModel, \
                  LearningToSimGaussianModel, SHiPModel, BernoulliModel, \
                  ModelDegenerate, ModelInstrict, \
                  RosenbrockModelInstrict, RosenbrockModelDegenerate, RosenbrockModelDegenerateInstrict
from ffjord_ensemble_model import FFJORDModel as FFJORDEnsembleModel
from ffjord_model import FFJORDModel
from gmm_model import GMMModel
from gan_model import GANModel
from linear_model import LinearModelOnPsi
from optimizer import *
from logger import SimpleLogger, CometLogger, GANLogger, RegressionLogger
from base_model import BaseConditionalGenerationOracle, ShiftedOracle
from constraints_utils import make_box_barriers, add_barriers_to_oracle
from experience_replay import ExperienceReplay, ExperienceReplayAdaptive
from adaptive_borders import AdaptiveBorders
REWEIGHT = False

if REWEIGHT:
    from hep_ml import reweight

from base_model import average_block_wise
from RegressionNN.regression_model import RegressionModel, RegressionRiskModel

def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(get_freer_gpu()))
else:
    device = torch.device('cpu')
print("Using device = {}".format(device))


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


def end_to_end_training(epochs: int,
                        model_cls: BaseConditionalGenerationOracle,
                        optimizer_cls: BaseOptimizer,
                        optimized_function_cls: BaseConditionalGenerationOracle,
                        logger: BaseLogger,
                        model_config: dict,
                        optimizer_config: dict,
                        n_samples_per_dim: int,
                        step_data_gen: float,
                        n_samples: int,
                        current_psi: Union[List[float], torch.tensor],
                        reuse_optimizer: bool = False,
                        reuse_model: bool = False,
                        shift_model: bool = False,
                        finetune_model: bool = False,
                        use_experience_replay: bool =True,
                        add_box_constraints: bool = False,
                        experiment = None,
                        use_adaptive_borders=True
                        ):
    """

    :param epochs: int
        number of local training steps to perfomr
    :param model_cls: BaseConditionalGenerationOracle
        model that is able to generate samples and calculate loss function
    :param optimizer_cls: BaseOptimizer
    :param logger: BaseLogger
    :param model_config: dict
    :param optimizer_config: dict
    :param n_samples_per_dim: int
    :param step_data_gen: float
    :param n_samples: int
    :param current_psi:
    :param reuse_model:
    :param reuse_optimizer:
    :param finetune_model:
    :param shift_model:

    :return:
    """
    #gan_logger = GANLogger(experiment)
    gan_logger = RegressionLogger(experiment)
    #print(optimizer_config['x_step'])
    y_sampler = optimized_function_cls(device=device, psi_init=current_psi)
    model = model_cls(y_model=y_sampler, **model_config, logger=gan_logger).to(device)
    optimizer = optimizer_cls(
        oracle=model,
        x=current_psi,
        **optimizer_config
    )
    print(model_config)
    exp_replay = ExperienceReplay(
        psi_dim=model_config['psi_dim'],
        y_dim=model_config['y_dim'],
        x_dim=model_config['x_dim'],
        device=device
    )
    if use_adaptive_borders:
        adaptive_border = AdaptiveBorders(psi_dim=model_config['psi_dim'], step=step_data_gen)
        exp_replay = ExperienceReplayAdaptive(
            psi_dim=model_config['psi_dim'],
            y_dim=model_config['y_dim'],
            x_dim=model_config['x_dim'],
            device=device
        )
    for epoch in range(epochs):
        # generate new data sample
        # condition
        if use_adaptive_borders:
            x, condition, conditions_grid, r_grid = y_sampler.generate_local_data_lhs_normal(
                n_samples_per_dim=n_samples_per_dim,
                sigma=adaptive_border.sigma,
                current_psi=current_psi,
                n_samples=n_samples)
            if use_experience_replay:
                x_exp_replay, condition_exp_replay = exp_replay.extract(psi=current_psi, sigma=adaptive_border.sigma)
                exp_replay.add(y=x, condition=condition)
                x = torch.cat([x, x_exp_replay], dim=0)
                condition = torch.cat([condition, condition_exp_replay], dim=0)
        else:
            x, condition = y_sampler.generate_local_data_lhs(
                n_samples_per_dim=n_samples_per_dim,
                step=step_data_gen,
                current_psi=current_psi,
                n_samples=n_samples)
        if use_experience_replay:
                x_exp_replay, condition_exp_replay = exp_replay.extract(psi=current_psi, step=step_data_gen)
                exp_replay.add(y=x, condition=condition)
                x = torch.cat([x, x_exp_replay], dim=0)
                condition = torch.cat([condition, condition_exp_replay], dim=0)
        if model_config["predict_risk"]:
            condition = condition[::n_samples_per_dim, :current_psi.shape[0]]
            x = y_sampler.func(condition, num_repetitions=n_samples_per_dim).reshape(-1, x.shape[1])
        print(x.shape, condition.shape)
        print(
            condition[:, :model_config['psi_dim']].std(dim=0).detach().cpu().numpy(),
            np.percentile(condition[:, :model_config['psi_dim']].detach().cpu().numpy(), q=[5, 95], axis=0)
        )

        model.train()
        if reuse_model:
            if shift_model:
                if isinstance(model, ShiftedOracle):
                    model.set_shift(current_psi.clone().detach())
                else:
                    model = ShiftedOracle(oracle=model, shift=current_psi.clone().detach())
                model.fit(x, condition=condition, weights=weights)
            else:
                model.fit(x, condition=condition, weights=weights)
        else:
            # if not reusing model
            # then at each epoch re-initialize and re-fit
            model = model_cls(y_model=y_sampler, **model_config, logger=gan_logger).to(device)
            model.fit(x, condition=condition, weights=weights)

        model.eval()

        if use_adaptive_borders:
            adaptive_border.step(model=model, conditions_grid=conditions_grid, r_grid=r_grid)

        if reuse_optimizer:
            optimizer.update(oracle=model,
                             x=current_psi)
        else:
            # find new psi
            optimizer = optimizer_cls(oracle=model,
                                      x=current_psi,
                                      **optimizer_config)

        if add_box_constraints:
            box_barriers = make_box_barriers(current_psi, step_data_gen)
            add_barriers_to_oracle(oracle=model, barriers=box_barriers)

        current_psi, status, history = optimizer.optimize()

        try:
            # logging optimization, i.e. statistics of psi
            logger.log_grads(model, y_sampler, current_psi, n_samples_per_dim)
            logger.log_performance(y_sampler=y_sampler,
                                   current_psi=current_psi,
                                   n_samples=n_samples)
            logger.log_optimizer(optimizer)
            if use_adaptive_borders:
                adaptive_border.log(experiment)
            # too long for ship...
            """
            if not isinstance(y_sampler, SHiPModel):
                logger.log_oracle(oracle=model,
                                  y_sampler=y_sampler,
                                  current_psi=current_psi,
                                  step_data_gen=step_data_gen,
                                  num_samples=200)
            """
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            # raise
        torch.cuda.empty_cache()
    return


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--optimizer', type=str, default='GradientDescentOptimizer')
@click.option('--logger', type=str, default='CometLogger')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--epochs', type=int, default=10000)
@click.option('--n_samples', type=int, default=10)
@click.option('--lr', type=float, default=1e-1)
@click.option('--step_data_gen', type=float, default=0.1)
@click.option('--n_samples_per_dim', type=int, default=3000)
@click.option('--reuse_optimizer', type=bool, default=False)
@click.option('--reuse_model', type=bool, default=False)
@click.option('--shift_model', type=bool, default=False)
@click.option('--finetune_model', type=bool, default=False)
@click.option('--add_box_constraints', type=bool, default=False)
@click.option('--use_experience_replay', type=bool, default=True)
@click.option('--use_adaptive_borders', type=bool, default=True)
@click.option('--init_psi', type=str, default="0., 0.")
def main(model,
         optimizer,
         logger,
         optimized_function,
         project_name,
         work_space,
         tags,
         model_config_file,
         optimizer_config_file,
         epochs,
         n_samples,
         step_data_gen,
         n_samples_per_dim,
         reuse_optimizer,
         reuse_model,
         shift_model,
         lr,
         finetune_model,
         use_experience_replay,
         add_box_constraints,
         use_adaptive_borders,
         init_psi
         ):
    model_config = getattr(__import__(model_config_file), 'model_config')
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)
    model_config['psi_dim'] = psi_dim
    optimizer_config['x_step'] = step_data_gen
    optimizer_config['lr'] = lr

    optimized_function_cls = str_to_class(optimized_function)
    model_cls = str_to_class(model)
    optimizer_cls = str_to_class(optimizer)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameter('model_type', model)
    experiment.log_parameter('optimizer_type', optimizer)
    experiment.log_parameters(
        {"model_{}".format(key): value for key, value in model_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('line_search_options', {}).items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('optim_params', {}).items()}
    )
    # experiment.log_asset("./gan_model.py", overwrite=True)
    # experiment.log_asset("./optim.py", overwrite=True)
    # experiment.log_asset("./train.py", overwrite=True)
    # experiment.log_asset("../model.py", overwrite=True)

    logger = str_to_class(logger)(experiment)
    print("Using device = {}".format(device))

    end_to_end_training(
        epochs=epochs,
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        optimized_function_cls=optimized_function_cls,
        logger=logger,
        model_config=model_config,
        optimizer_config=optimizer_config,
        current_psi=init_psi,
        n_samples_per_dim=n_samples_per_dim,
        step_data_gen=step_data_gen,
        n_samples=n_samples,
        reuse_optimizer=reuse_optimizer,
        reuse_model=reuse_model,
        shift_model=shift_model,
        finetune_model=finetune_model,
        add_box_constraints=add_box_constraints,
        use_experience_replay=use_experience_replay,
        experiment=experiment,
        use_adaptive_borders=use_adaptive_borders
    )


if __name__ == "__main__":
    main()

