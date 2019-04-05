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
from gqn_dataset import GQNDataset, Scene, transform_viewpoint, sample_batch
from scheduler import AnnealingStepLR
from model import NSG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    parser.add_argument('--dataset', type=str, default='Shepard-Metzler', help='dataset (dafault: Shepard-Mtzler)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/workspace/dataset/shepard_metzler_7_parts-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/workspace/dataset/shepard_metzler_7_parts-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/workspace/logs')
    parser.add_argument('--log_dir', type=str, help='log directory (default: NSG)', default='nsg')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=10000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--representation', type=str, help='representation network (default: pool)', default='tower')
    parser.add_argument('--layers', type=int, help='number of generative layers (default: 12)', default=12)
    parser.add_argument('--shared_core', type=bool, \
                        help='whether to share the weights of the cores across generation steps (default: True)', \
                        default=True)
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--M', type=int, help='M in test', default=3)
    parser.add_argument('--num_epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
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
    os.mkdir(os.path.join(log_dir,'runs'))

    # TensorBoardX
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    # Dataset
    train_dataset = GQNDataset(root_dir=train_data_dir, target_transform=transform_viewpoint)
    test_dataset = GQNDataset(root_dir=test_data_dir, target_transform=transform_viewpoint)
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
    model = NSG(L=L, shared_core=args.shared_core, z_dim=args.z_dim).to(device)

    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
#     optimizer = torch.optim.Adam(model.parameters())

    scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

#     train_iter = iter(train_loader)
    x_data_test, v_data_test = next(iter(test_loader))

    step = 0
    # Training Iterations
    for epoch in range(args.num_epoch):
        if len(args.device_ids)>1:
            model.module.sigma.param.requires_grad = True if epoch>10 else False
        else:
            model.sigma.param.requires_grad = True if epoch>10 else False
            
        for t, (x_data, v_data) in enumerate(tqdm(train_loader)):
            model.train()
            x_data = x_data.to(device)
            v_data = v_data.to(device)
            train_elbo, train_nll, train_kl = model(x_data, v_data, D)

            # Compute empirical ELBO gradients
            train_elbo.mean().backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Update optimizer state
            scheduler.step()

            # Logs
            writer.add_scalar('train_elbo', train_elbo.mean(), step)
            writer.add_scalar('train_nll', train_nll.mean(), step)
            writer.add_scalar('train_kl', train_kl.mean(), step)

            with torch.no_grad():
                model.eval()
                # Write logs to TensorBoard
                if step % log_interval_num == 0:
                    x_data_test = x_data_test.to(device)
                    v_data_test = v_data_test.to(device)

                    test_elbo, test_nll, test_kl = model(x_data_test, v_data_test, D)

                    context_idx = random.sample(range(x_data_test.size(1)), M)
                    sample_idx = random.sample(range(x_data_test.size(1)), 36)
                    if len(args.device_ids)>1:
                        x_gen = model.module.generate(v_data_test[:, sample_idx])
                        x_pred = model.module.predict(x_data_test[:, context_idx], v_data_test[:, context_idx], v_data_test[:, sample_idx])
                    else:
                        x_gen = model.generate(v_data_test[:, sample_idx])
                        x_pred = model.predict(x_data_test[:, context_idx], v_data_test[:, context_idx], v_data_test[:, sample_idx])
            
                    writer.add_scalar('test_elbo', test_elbo.mean(), step)
                    writer.add_scalar('test_nll', test_nll.mean(), step)
                    writer.add_scalar('test_kl', test_kl.mean(), step)
                    writer.add_image('test_ground_truth', make_grid(x_data_test[0, sample_idx], 6, pad_value=1), step)
                    writer.add_image('test_prediction', make_grid(x_pred[0], 6, pad_value=1), step)
                    writer.add_image('test_generation', make_grid(x_gen[0], 6, pad_value=1), step)

                if step % save_interval_num == 0:
                    state_dict = model.module.state_dict() if len(args.device_ids)>1 else model.state_dict()
                    torch.save(state_dict, log_dir + "/models/model-{}.pt".format(step))
                
            step += 1
                
    state_dict = model.module.state_dict() if len(args.device_ids)>1 else model.state_dict()
    torch.save(state_dict, log_dir + "/models/model-final.pt")  
    writer.close()
