#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
import datetime
import torch
torch.set_num_threads(1)
import pickle 

from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import D2STGNN
import yaml
import setproctitle
import logging

def dataloaderEveryYears(dataset_name, load_pkl, data_dir, config):
    if load_pkl:
        t1   = time.time()
        dataloader  = pickle.load(open('./output/dataloader_' + dataset_name + '.pkl', 'rb'))
        t2  = time.time()
        logging.info("Load dataset: {:.2f}s...".format(t2-t1))
    else:
        t1   = time.time()
        batch_size  = config['model_args']['batch_size']
        dataloader  = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        pickle.dump(dataloader, open('./output/dataloader_' + dataset_name + '.pkl', 'wb'))
        t2  = time.time()
        logging.info("Load dataset: {:.2f}s...".format(t2-t1))
    scaler          = dataloader['scaler']
    
    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':  # traffic flow
        _min = pickle.load(open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None
    
    t1   = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'],
             is_npz=config['data_args']['is_npz'])
    t2  = time.time()
    logging.info("Load adjacent matrix: {:.2f}s...".format(t2-t1))
    return dataloader, scaler, _min, _max, adj_mx, adj_ori

def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BAST', help='Dataset name.')
    parser.add_argument('--stream', type=int, default=0, help='Dataset name.')
    args = parser.parse_args()
    config_path = "configs/" + args.dataset + ".yaml"

    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        
    data_dir        = config['data_args']['data_dir']
    dataset_name    = config['data_args']['data_dir'].split("/")[-1]

    device          = torch.device(config['start_up']['device'])
    timestr = '{0:_%Y-%m-%d__%H_%M_%S}'.format(datetime.datetime.now())
    save_path       = './output/' + config['start_up']['model_name'] + "_" + dataset_name + timestr + ".pt"             # the best model
    save_path_resume= './output/' + config['start_up']['model_name'] + "_" + dataset_name + timestr + "_resume.pt"      # the resume model
    save_path_logger= './output/' + config['start_up']['model_name'] + "_" + dataset_name + timestr + ".log"    
    logging.basicConfig(filename=save_path_logger, level=logging.INFO)  
    load_pkl        = config['start_up']['load_pkl']
    model_name      = config['start_up']['model_name']

    model_name      = config['start_up']['model_name']
    begin_year      = config['start_up']['begin_year']
    end_year        = config['start_up']['end_year']
    setproctitle.setproctitle("{0}.{1}@S22".format(model_name, dataset_name))
# ================================ Hyper Parameters ================================= #
    # model parameters
    model_args  = config['model_args']
    model_args['device']        = device
    model_args['num_nodes']     = adj_mx[0].shape[0]
    model_args['adjs']          = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori']      = torch.tensor(adj_ori).to(device)
    model_args['dataset']       = dataset_name

    
# ============================= Model =========================================================== #
    # log
    logger  = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    # init the model
    model   = D2STGNN(**model_args).to(device)

    # get a trainer
    engine  = trainer(scaler, model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # begin training:
    train_time  = []    # training time
    val_time    = []    # validate time
# ========================== load dataset, adjacent matrix, node embeddings ====================== #
    for i in range(begin_year, end_year+1):
        dataloader, scaler, _min, _max, adj_mx, adj_ori = dataloaderEveryYears(dataset_name=dataset_name, 
                                    load_pkl=load_pkl, data_dir=data_dir, config=config)
        # training strategy parametes
        optim_args                  = config['optim_args']
        optim_args['cl_steps']      = optim_args['cl_epochs'] * len(dataloader['train_loader'])
        optim_args['warm_steps']    = optim_args['warm_epochs'] * len(dataloader['train_loader'])

        logging.info("Whole trainining iteration is " + str(len(dataloader['train_loader'])))

# ========================== Train ============================================================== #

        # training init: resume model & load parameters
        mode = config['start_up']['mode']
        assert mode in ['test', 'resume', 'scratch']
        resume_epoch = 0
        if mode == 'test':
            model = load_model(model, save_path)        # resume best
        else:
            if mode == 'resume':
                resume_epoch = config['start_up']['resume_epoch']
                model = load_model(model, save_path_resume)
            else:       # scratch
                resume_epoch = 0
        
        batch_num   = resume_epoch * len(dataloader['train_loader'])     # batch number (maybe used in schedule sampling)

        engine.set_resume_lr_and_cl(resume_epoch, batch_num)
# =============================================================== Training ================================================================= #   
        if mode != 'test':
            for epoch in range(resume_epoch + 1, optim_args['epochs']):
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                # train a epoch
                time_train_start    = time.time()

                current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
                train_loss = []
                train_mape = []
                train_rmse = []
                dataloader['train_loader'].shuffle()    # traing data shuffle when starting a new epoch.
                totaliter = 0
                avgmae = 0.0
                for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                    totaliter += 1
                    trainx          = data_reshaper(x, device)
                    trainy          = data_reshaper(y, device)
                    mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
                    # mae, mape, rmse = 0,0,0
                    avgmae += mae
                    train_loss.append(mae)
                    train_mape.append(mape)
                    train_rmse.append(rmse)
                    batch_num += 1
                logging.info("train : {0}: {1}".format(epoch, avgmae/totaliter))
                time_train_end      = time.time()
                train_time.append(time_train_end - time_train_start)

                current_learning_rate = engine.optimizer.param_groups[0]['lr']

                if engine.if_lr_scheduler:
                    engine.lr_scheduler.step()
                # record history loss
                mtrain_loss = np.mean(train_loss)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)
# =============================================================== Validation ================================================================= #
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                time_val_start      = time.time()
                mvalid_loss, mvalid_mape, mvalid_rmse, = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
                # mvalid_loss, mvalid_mape, mvalid_rmse, = 0,0,0
                time_val_end        = time.time()
                val_time.append(time_val_end - time_val_start)

                curr_time   = str(time.strftime("%d-%H-%M", time.localtime()))
                log = 'Current Time: ' + curr_time + ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} | Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAPE: {:.4f} | LR: {:.6f}'
                logging.info(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mape, current_learning_rate))
                early_stopping(mvalid_loss, engine.model)
                if early_stopping.early_stop:
                    logging.info('Early stopping!')
                    break
# =============================================================== Test ================================================================= #
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                engine.test(model, save_path_resume, device, dataloader, scaler, model_name, _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)

            logging.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            logging.info("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
        else:
            engine.test(model, save_path_resume, device, dataloader, scaler, model_name, save=False, _max=_max, _min=_min, loss=engine.loss, dataset_name=dataset_name)

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end   = time.time()
    logging.info("Total time spent: {0}".format(t_end - t_start))
