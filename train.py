import argparse
import datetime
import math
import os
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataset import GQNDataset, SceneNet, Scene, transform_viewpoint, sample_batch
from scheduler import AnnealingStepLR
from model import NSG, GQN, CGQN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--dataset', type=str, default='Shepard-Metzler', help='dataset (dafault: Shepard-Metzler)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/workspace/data/GQN/shepard_metzler_7_parts-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/workspace/data/GQN/shepard_metzler_7_parts-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs')
    parser.add_argument('--log_dir', type=str, help='log directory (default: NSG)', default='NSG')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=10000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--representation', type=str, help='representation network (default: tower)', default='tower')
    parser.add_argument('--layers', type=int, help='number of generative layers (default: 12)', default=12)
    parser.add_argument('--shared_core', type=bool, \
                        help='whether to share the weights of the cores across generation steps (default: True)', \
                        default=True)
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--v_dim', type=int, default=5)
    parser.add_argument('--M', type=int, help='M in test', default=3)
    parser.add_argument('--num_epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
    parser.add_argument('--model', type=str, help='which model to use (default: NSG)', default='NSG')
    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    os.mkdir(log_dir)
    os.mkdir(os.path.join(log_dir, 'models'))
    os.mkdir(os.path.join(log_dir, 'optimizers'))
    
    os.mkdir(os.path.join(log_dir,'runs'))

    # TensorBoardX
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    # Dataset
    train_dataset = GQNDataset(root_dir=train_data_dir)
    test_dataset = GQNDataset(root_dir=test_data_dir)
    D = args.dataset
    
    # Pixel standard-deviation
#     sigma_i, sigma_f = 2.0, 0.7
#     sigma = sigma_i

    # Number of scenes over which each weight update is computed
    B = args.batch_size
    
    M = args.M
    
    # Number of generative layers
    L =args.layers

    # Maximum number of training steps
    S_max = args.gradient_steps

    # Define model
    if args.model=='NSG':
        model = NSG(L=L, shared_core=args.shared_core, z_dim=args.z_dim, v_dim=args.v_dim).to(device)
    elif args.model=='GQN':
        model = GQN(L=L, shared_core=args.shared_core, z_dim=args.z_dim, v_dim=args.v_dim).to(device)
    elif args.model=='CGQN':
        model = CGQN(L=L, shared_core=args.shared_core, z_dim=args.z_dim, v_dim=args.v_dim).to(device)

    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

#     optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

#     optimizer = torch.optim.Adam(model.parameters())

#     scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

#     train_iter = iter(train_loader)
    x_data_test, v_data_test = next(iter(test_loader))

    step = 0
    # Training Iterations
    for epoch in range(args.num_epoch):
#         if len(args.device_ids)>1:
#             model.module.sigma.param.requires_grad = True if epoch>=args.num_epoch//10 else False
#         else:
#             model.sigma.param.requires_grad = True if epoch>=args.num_epoch//10 else False
            
        for t, (x_data, v_data) in enumerate(tqdm(train_loader)):
            model.train()
            x, v = sample_batch(x_data, v_data, D)
            x, v = x.to(device), v.to(device)
            if args.model == "GQN" or args.model == "CGQN":
                train_elbo, train_nll, train_kl = model(x, v, True)
            else:
                train_elbo, train_nll, train_kl = model(x, v)

            # Compute empirical ELBO gradients
            train_elbo.mean().backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update optimizer state
#             scheduler.step()

            # Logs
            writer.add_scalar('train_elbo', train_elbo.mean(), step)
            writer.add_scalar('train_nll', train_nll.mean(), step)
            writer.add_scalar('train_kl', train_kl.mean(), step)

            with torch.no_grad():
                model.eval()
                # Write logs to TensorBoard
                if step % log_interval_num == 0:
                    x_test, v_test = sample_batch(x_data_test, v_data_test, D, seed=0)
                    x_test, v_test = x_test.to(device), v_test.to(device)
                    
                    if args.model == "GQN" or args.model == "CGQN":
                        test_elbo, test_nll, test_kl = model(x_test, v_test, False)  
                    else:
                        test_elbo, test_nll, test_kl = model(x_test, v_test)

                    random.seed(0)
                    context_idx = random.sample(range(x_test.size(1)), M)
                    x_context, v_context = x_test[:, context_idx], v_test[:, context_idx]
                    if len(args.device_ids)>1:
                        x_gen = model.module.generate(v_test)
                        x_pred = model.module.predict(x_context, v_context, v_test)
                        x_rec = model.module.reconstruct(x_test, v_test)
                    else:
                        x_gen = model.generate(v_test)
                        x_pred = model.predict(x_context, v_context, v_test)
                        x_rec = model.reconstruct(x_test, v_test)
            
                    writer.add_scalar('test_elbo', test_elbo.mean(), step)
                    writer.add_scalar('test_nll', test_nll.mean(), step)
                    writer.add_scalar('test_kl', test_kl.mean(), step)
                    writer.add_image('test_context', make_grid(x_context[0], 5, pad_value=1), step)
                    writer.add_image('test_ground_truth', make_grid(x_test[0], 5, pad_value=1), step)
                    writer.add_image('test_prediction', make_grid(x_pred[0], 5, pad_value=1), step)
                    writer.add_image('test_generation', make_grid(x_gen[0], 5, pad_value=1), step)
                    writer.add_image('test_reconstruction', make_grid(x_rec[0], 5, pad_value=1), step)

            step += 1
            
        state_dict = model.module.state_dict() if len(args.device_ids)>1 else model.state_dict()
        torch.save(state_dict, log_dir + f"/models/model-{epoch}.pt")
        
        optimizer_state_dict = optimizer.state_dict()
        torch.save(optimizer_state_dict, log_dir + f"/optimizers/optimizer-{epoch}.pt")
                
    state_dict = model.module.state_dict() if len(args.device_ids)>1 else model.state_dict()
    torch.save(state_dict, log_dir + "/models/model-final.pt")  
    writer.close()
