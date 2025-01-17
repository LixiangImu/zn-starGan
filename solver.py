from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import cv2
from metrics import calculate_metrics


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, voc_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.voc_loader = voc_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        if self.use_tensorboard:
            try:
                from logger import Logger
                self.logger = Logger(self.log_dir)
            except ImportError:
                print("TensorFlow not found - running without tensorboard logging")
                self.use_tensorboard = False

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        if dataset == 'VOC':
            c_trg_list = []
            c_trg = 1 - c_org  # 直接取反原始标签
            c_trg_list.append(c_trg.to(self.device))
            return c_trg_list
        
        # 原有的CelebA和RaFD的代码保持不变
        elif dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

            c_trg_list = []
            for i in range(c_dim):
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg = c_org.clone()
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg = (c_trg[:, i] == 0)  # Reverse attribute value.
                c_trg_list.append(c_trg.to(self.device))
            return c_trg_list
        elif dataset == 'RaFD':
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list = []
            for i in range(c_dim):
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
                c_trg_list.append(c_trg.to(self.device))
            return c_trg_list

    def classification_loss(self, logit, target, dataset='VOC'):
        """Compute binary cross entropy loss."""
        if dataset == 'VOC':
            return F.binary_cross_entropy_with_logits(logit, target.float())
        else:
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # 设置数据加载器
        data_loader = self.voc_loader
        
        # 初始化学习率
        g_lr = self.g_lr
        d_lr = self.d_lr
        
        # 获取固定的输入用于调试
        data_iter = iter(data_loader)
        x_fixed, _, c_org, _ = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_trg = torch.ones_like(c_org).to(self.device)

        # 开始训练
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):
            try:
                x_real_A, x_real_B, label_org, label_trg = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_A, x_real_B, label_org, label_trg = next(data_iter)

            # 准备输入，确保所有张量都在同一设备上
            x_real_A = x_real_A.to(self.device)     # trainA的图片
            x_real_B = x_real_B.to(self.device)     # trainB的图片
            label_org = label_org.to(self.device)   # trainA的标签 (0)
            label_trg = label_trg.to(self.device)   # trainB的标签 (1)
            
            # =================================================================================== #
            #                             1. 训练判别器                                            #
            # =================================================================================== #
            
            # 计算真实图像的判别器损失（使用trainB的图片）
            out_src, out_cls = self.D(x_real_B)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

            # 计算生成图像的判别器损失（将trainA的图片转换为trainB的风格）
            x_fake = self.G(x_real_A, label_trg)
            out_src, _ = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # 计算梯度惩罚
            alpha = torch.rand(x_real_A.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real_B.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # 判别器的总损失
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp + self.lambda_cls * d_loss_cls

            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # =================================================================================== #
            #                             2. 训练生成器                                            #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # 原始域到目标域的转换
                x_fake = self.G(x_real_A, label_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # 重构损失（确保内容保持不变）
                x_reconst = self.G(x_fake, label_org)
                g_loss_rec = torch.mean(torch.abs(x_real_A - x_reconst))

                # 添加内容保持损失（L1距离）
                g_loss_content = torch.mean(torch.abs(x_real_A - x_fake))

                # 生成器的总损失
                g_loss = g_loss_fake + \
                        self.lambda_rec * g_loss_rec + \
                        self.lambda_cls * g_loss_cls + \
                        0.5 * g_loss_content  # 添加内容损失

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            # =================================================================================== #
            #                             3. 记录和打印训练信息                                    #
            # =================================================================================== #
            
            # 打印训练信息
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print(f"Elapsed [{et}], Iteration [{i+1}/{self.num_iters}]")
                print(f"D/loss_real: {d_loss_real.item():.4f}, D/loss_fake: {d_loss_fake.item():.4f}")
                print(f"D/loss_cls: {d_loss_cls.item():.4f}, D/loss_gp: {d_loss_gp.item():.4f}")
                print(f"G/loss_fake: {g_loss_fake.item():.4f}, G/loss_rec: {g_loss_rec.item():.4f}")
                print(f"G/loss_cls: {g_loss_cls.item():.4f}")

            # 保存生成的样本
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake = self.G(x_fixed, c_trg)
                    x_concat = torch.cat([x_fixed, x_fake], dim=3)
                    sample_path = os.path.join(self.sample_dir, f'{i+1}-images.jpg')
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print(f'Saved real and fake images into {sample_path}...')

            # 保存模型
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, f'{i+1}-G.ckpt')
                D_path = os.path.join(self.model_save_dir, f'{i+1}-D.ckpt')
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print(f'Saved model checkpoints into {self.model_save_dir}...')

            # Decay learning rates
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.voc_loader
        
        with torch.no_grad():
            for i, (x_real_A, _, _, _) in enumerate(data_loader):
                # 准备输入
                x_real_A = x_real_A.to(self.device)
                c_trg = torch.ones(x_real_A.size(0), 1).to(self.device)  # 目标域标签设为1（trainB的风格）

                # 生成转换后的图片
                x_fake = self.G(x_real_A, c_trg)
                
                # 保存原始图片和转换后的图片
                x_concat = torch.cat([x_real_A, x_fake], dim=3)
                result_path = os.path.join(self.result_dir, f'{i+1}-images.jpg')
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print(f'Saved real and fake images into {result_path}...')

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def evaluate(self):
        """评估模型性能"""
        self.restore_model(self.test_iters)  # 加载训练好的模型
        self.G.eval()
        
        all_predictions = []
        all_targets = []
        
        print('开始评估...')
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.voc_loader):  # 修改这里，只接收两个值
                # 准备数据
                x_real = x_real.to(self.device)
                
                # 生成目标域标签（全1）
                c_trg = torch.ones_like(c_org).to(self.device)
                
                # 生成图像
                x_fake = self.G(x_real, c_trg)
                
                # 将生成的图像转换为二值图像
                pred = self.denorm(x_fake).cpu().numpy()
                target = self.denorm(x_real).cpu().numpy()
                
                # 转换为灰度图并二值化
                pred = (pred.mean(axis=1) > 0.5).astype(np.float32)
                target = (target.mean(axis=1) > 0.5).astype(np.float32)
                
                all_predictions.append(pred)
                all_targets.append(target)
                
                if i % 10 == 0:
                    print(f'处理批次: [{i}/{len(self.voc_loader)}]')
        
        # 合并所有批次的结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 计算指标
        metrics = calculate_metrics(predictions, targets)
        
        # 打印结果
        print('\n评估结果:')
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"IoU Score: {metrics['iou']:.4f}")
        
        # 保存结果到文件
        result_path = os.path.join(self.result_dir, 'evaluation_results.txt')
        with open(result_path, 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f'{metric_name}: {value:.4f}\n')
        
        print(f'\n评估结果已保存到: {result_path}')