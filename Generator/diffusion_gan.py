#diffusion_gan

import os
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import glob
import argparse

from Generator.torch_utils.torch_utils import *
from Generator.torch_utils.utils import *
import Generator.torch_utils.language_helpers
from Generator.networks import *
from Generator.diffusion import Diffusion

plt.switch_backend('agg')


class GAN():
    def __init__(
            self,
            batch_size=64,
            lr=0.0001,
            # num_epochs=80,
            seq_len=80,
            data_dir='/Data/SC/SC_seq_short.txt',
            run_name='test',
            hidden=512,
            d_steps=10,
            gain=1,
    ):
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.gain = gain
        self.g_steps = 1
        self.ada_interval = 4
        self.ada_target = 0.6
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        self.sample_dir = './samples/' + run_name + "/"
        self.load_data(data_dir)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        torch.cuda.set_device(2)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.build_model()
        self.diffusion = Diffusion()

    def build_model(self):
        self.G = Generator_net(len(self.charmap), self.seq_len, self.batch_size, self.hidden).to(self.device)
        self.D = Discriminator_net(len(self.charmap), self.seq_len, self.batch_size, self.hidden).to(self.device)
        print(self.G)
        print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def load_data(self, datadir):
        max_examples = 1e6
        lines, self.charmap, self.inv_charmap = Generator.utils.language_helpers.load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.data = lines

        sample_total = len(self.data)  # Total sample size of the dataset
        return sample_total

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + "G_weights_{}.pth".format(epoch))
        torch.save(self.D.state_dict(), self.checkpoint_dir + "D_weights_{}.pth".format(epoch))

    def load_model(self, directory=''):
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("[*] Checkpoint not found! Starting from scratch.")
            return 1
        G_file = max(list_G, key=os.path.getctime)
        D_file = max(list_D, key=os.path.getctime)
        epoch_found = int((G_file.split('_')[-1]).split('.')[0])
        print("[*] Checkpoint {} found at {}!".format(epoch_found, directory))
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))
        return epoch_found

    def disc_train_iteration(self, real_data):
        self.D_optimizer.zero_grad()
        fake_data = self.sample_generator(self.batch_size)

        steps = self.diffusion.t_epl[:32]
        real_noisy_samples = self.diffusion.get_noisy_samples(real_data, steps)
        fake_noisy_samples = self.diffusion.get_noisy_samples(fake_data, steps)

        d_real_errs, d_fake_errs = [], []
        real_preds = []

        for t in range(len(real_noisy_samples)):
            d_real_pred = self.D(real_noisy_samples[t])
            d_fake_pred = self.D(fake_noisy_samples[t])

            d_real_err = torch.nn.functional.softplus(-d_real_pred)
            d_fake_err = torch.nn.functional.softplus(d_fake_pred)

            d_real_errs.append(d_real_err)
            d_fake_errs.append(d_fake_err)
            real_preds.append(d_real_pred)

        d_real_err = torch.mean(torch.stack(d_real_errs))
        d_fake_err = torch.mean(torch.stack(d_fake_errs))

        d_err = d_real_err + d_fake_err
        d_err.mean().backward()
        self.D_optimizer.step()

        real_preds = torch.cat(real_preds)
        r_d = torch.sign(torch.sigmoid(real_preds) - 0.5).mean().item()

        return d_fake_err.mean().item(), d_real_err.mean().item(), d_err.mean().item(), r_d

    def sample_generator(self, num_sample):
        z_input = Variable(torch.randn(num_sample, 128)).to(self.device)
        generated_data = self.G(z_input)
        return generated_data

    def gen_train_iteration(self):
        self.G.zero_grad()
        z_input = Variable(torch.randn(self.batch_size, 128)).to(self.device)
        g_fake_data = self.G(z_input)
        dg_fake_pred = self.D(g_fake_data)
        g_err = torch.nn.functional.softplus(-dg_fake_pred)
        g_err.mean().backward()
        self.G_optimizer.step()
        return g_err.mean().item()

    def train_model(self, load_dir):  # Model training
        total_iterations = 3000  # Number of iterations
        losses_f = open(self.checkpoint_dir + "losses.txt", 'a+')
        d_fake_losses, d_real_losses = [], []
        G_losses, D_losses = [], []

        # one-hot
        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        sample_total = len(self.data)

        counter = 0
        n_batches = int(len(self.data) / self.batch_size)
        while counter < total_iterations:
            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in
                     self.data[idx * self.batch_size:(idx + 1) * self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1,
                                                                                         len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)

                # Train the discriminator
                d_fake_err, d_real_err, d_err, r_d = self.disc_train_iteration(real_data)

                d_real_losses.append(d_real_err)
                d_fake_losses.append(d_fake_err)
                D_losses.append(d_err)

                # Update T every four iterations
                if counter % self.ada_interval == 0:
                    C = (self.batch_size * self.ada_interval) / sample_total
                    adjust = np.sign(r_d - self.ada_target) * C
                    self.diffusion.p = (self.diffusion.p + adjust).clip(min=0., max=1.)
                    self.diffusion.update_T()

                # If the current number of iterations is a multiple of D training steps (10),
                # then perform G training (train G and return the G loss)
                if counter % self.d_steps == 0:
                    g_err = self.gen_train_iteration()

                if counter % 100 == 99:
                    self.save_model(counter)

                if counter % 10 == 9:
                    self.sample(counter)
                    summary_str = 'Iteration [{}/{}] loss_d: {}, loss_g: {}'.format(counter, total_iterations, d_err,
                                                                                    g_err)
                    print(summary_str)
                    losses_f.write(summary_str)

                counter += 1
                if counter >= total_iterations:  # Exit the loop when the total number of iterations is reached
                    break

    def sample(self, epoch):
        z = Variable(torch.randn(self.batch_size, 128)).to(self.device)
        self.G.eval()
        torch_seqs = self.G(z)
        seqs = (torch_seqs.data).cpu().numpy()
        decoded_seqs = [decode_one_seq(seq, self.inv_charmap) + "\n" for seq in seqs]
        with open(self.sample_dir + "sampled_{}.txt".format(epoch), 'w+') as f:
            f.writelines(decoded_seqs)
        self.G.train()


def main():
    parser = argparse.ArgumentParser(description='GAN for producing gene sequences.')
    parser.add_argument("--run_name", default="SC_1w_short_t5000_net1", help="Name for output files (checkpoint and sample dir)")
    parser.add_argument("--load_dir", default="",
                        help="Option to load checkpoint from other model (Defaults to run name)")
    args = parser.parse_args()
    model = GAN(run_name=args.run_name)
    model.train_model(args.load_dir)


if __name__ == '__main__':
    main()