import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset

import pandas as pd 
import os 

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10','svhn','mvtec', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp','mvtec_LeNet',
                                               'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
                                               'thyroid_mlp']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for Deep SAD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SAD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=0,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')

@click.option('--eps', type=float, default=8/255)            
@click.option('--alpha', type=float, default=1e-2)  

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, eta,
         ratio_known_normal, ratio_known_outlier, ratio_pollution, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay,
         pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes,eps,alpha):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(cfg.settings['eta'])
    deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         optimizer_name=cfg.settings['ae_optimizer_name'],
                         lr=cfg.settings['ae_lr'],
                         n_epochs=cfg.settings['ae_n_epochs'],
                         lr_milestones=cfg.settings['ae_lr_milestone'],
                         batch_size=cfg.settings['ae_batch_size'],
                         weight_decay=cfg.settings['ae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deepSAD.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    
    mine_result = {}
    mine_result['Attack_Type'] = []
    mine_result['Attack_Target'] = []
    mine_result['ADV_AUC'] = []   
    mine_result['setting'] = [] 
    
    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,attack_type='fgsm',epsilon=cfg.settings['eps'],alpha=cfg.settings['alpha'])
    
    
    clear_auc=deepSAD.results['clear_auc']
    normal_auc=deepSAD.results['normal_auc']
    anomal_auc=deepSAD.results['anomal_auc']
    both_auc=deepSAD.results['both_auc']
    
    print(f'FGSM Adv Adverserial Clean: {clear_auc}')
    print(f'FGSM Adv Adverserial Normal: {normal_auc}')
    print(f'FGSM Adv Adverserial Anomal: {anomal_auc}')
    print(f'FGSM Adv Adverserial Both: {both_auc}\n\n')
    
    mine_result['Attack_Type'].extend(['fgsm','fgsm','fgsm','fgsm'])
    mine_result['Attack_Target'].extend(['clean','normal','anomal','both'])
    mine_result['ADV_AUC'].extend([clear_auc,normal_auc,anomal_auc,both_auc])
    mine_result['setting'].extend([{'Dataset Name': dataset_name},{'Epsilon': cfg.settings['eps']},{'Alpha': cfg.settings['alpha']},{'Num Epoches': cfg.settings['n_epochs']}])        




    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,attack_type='pgd',epsilon=cfg.settings['eps'],alpha=cfg.settings['alpha'])
    clear_auc=deepSAD.results['clear_auc']
    normal_auc=deepSAD.results['normal_auc']
    anomal_auc=deepSAD.results['anomal_auc']
    both_auc=deepSAD.results['both_auc']

    mine_result['Attack_Type'].extend(['PGD','PGD','PGD','PGD'])
    mine_result['Attack_Target'].extend(['clean','normal','anomal','both'])
    mine_result['ADV_AUC'].extend([clear_auc,normal_auc,anomal_auc,both_auc])        
    mine_result['setting'].extend([{'Dataset Name': dataset_name},{'Epsilon': cfg.settings['eps']},{'Alpha': cfg.settings['alpha']},{'Num Epoches': cfg.settings['n_epochs']}])        
    
    print(f'PGD Adv Adverserial Clean: {clear_auc}')
    print(f'PGD Adv Adverserial Normal: {normal_auc}')
    print(f'PGD Adv Adverserial Anomal: {anomal_auc}')
    print(f'PGD Adv Adverserial Both: {both_auc}\n\n')

    df = pd.DataFrame(mine_result)    
    df.to_csv(os.path.join('./',f'Results_SAD_{dataset_name}_Class_{normal_class}.csv'), index=False)

    

    # Save results, model, and configuration
    deepSAD.save_results(export_json=xp_path + '/results.json')
    deepSAD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')




if __name__ == '__main__':
    main()