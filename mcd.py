import torch

class MCD:
    def __init__(self, model, optimizer, epochs, sub_window_num, n, k, eps_small, eps_big, temperature, lamb, percentile, device):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.sub_window_num = sub_window_num
        self.n = n
        self.k = k
        self.eps_small = eps_small
        self.eps_big = eps_big
        self.temperature = temperature
        self.lamb = lamb
        self.percentile = percentile
        self.device = device

    def generate_samples(self, sub_win_data):
        sub_win_size = sub_win_data.size(0)
        indices = torch.randint(0, sub_win_size, (self.n * self.k,))
        samples = sub_win_data[indices].clone().detach().requires_grad_(True)
        samples = samples.view(self.k, self.n, -1)
        return samples.to(self.device)

    def compute_positive_loss(self, embeddings):
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        norm_diff = torch.norm(diff, p=2, dim=2)
        mask = torch.triu(torch.ones_like(norm_diff), diagonal=1).bool()
        masked_norm_diff = norm_diff.masked_select(mask)
        mean_loss = masked_norm_diff.mean()
        return mean_loss, masked_norm_diff

    def compute_negative_loss(self, embeddings1, embeddings2):
        diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)
        norm_diff = torch.norm(diff, p=2, dim=2)
        mask = torch.eye(norm_diff.size(0), dtype=torch.bool, device=norm_diff.device)
        masked_norm_diff = norm_diff.masked_select(~mask)
        mean_loss = masked_norm_diff.mean()
        return mean_loss

    def contrastive_loss_function(self, pos_losses, neg_losses, temperature):
        """
        like InfoNCE
        """
        exp_pos_losses = torch.exp(torch.stack(pos_losses)/ temperature)
        exp_neg_losses = torch.exp(torch.stack(neg_losses)/ temperature)
        numerator = torch.sum(exp_pos_losses)
        denominator = numerator + torch.sum(exp_neg_losses)
        loss = torch.log(numerator / denominator)
        return loss

    def gradient_penalty(self, samples, output):
        gradients = torch.autograd.grad(outputs=output, inputs=samples,
                                    grad_outputs=torch.ones_like(output),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        penalty = ((gradients_norm - 1) ** 2).mean()
        return penalty

    def train(self, window_data):
        windows = torch.split(window_data, int(window_data.size(0) / self.sub_window_num))
        sub_win_samples = [self.generate_samples(sub_win) for sub_win in windows]

        for epoch in range(self.epochs):
            pos_losses = []
            weak_neg_losses = []

            for samples in sub_win_samples:
                # Positive sample pairs
                samples = samples.to(self.device)
                # h_p1 and h_p2
                embeddings = self.model(samples)
                embeddings_mean = embeddings.mean(dim=1)  

                # Loss for positive sample pairs
                pos_loss,_ = self.compute_positive_loss(embeddings_mean)
                pos_losses.append(pos_loss)

                # Weak negative sample pairs
                unchanged_samples = samples[:self.k//2]
                altered_samples = samples[self.k//2:].clone() + torch.normal(mean=0, std=self.eps_small, size=samples[self.k//2:].shape).to(self.device)

                # h_wn1 and h_wn2
                embeddings_unchanged = self.model(unchanged_samples).mean(dim=1)
                embeddings_altered = self.model(altered_samples).mean(dim=1)

                # Loss for weak negative sample pairs
                weak_neg_loss = self.compute_negative_loss(embeddings_unchanged, embeddings_altered)
                weak_neg_losses.append(weak_neg_loss)

            # Strong negative sample pairs
            # sub-window 1 and sub-window N_sub
            first_sub_win_samples = sub_win_samples[0]
            last_sub_win_samples = sub_win_samples[-1]
            last_sub_win_samples_altered = last_sub_win_samples.clone() + torch.normal(mean=0, std=self.eps_big, size=last_sub_win_samples.shape).to(self.device)

            # h_sn1 and h_sn2
            embeddings_first = self.model(first_sub_win_samples).mean(dim=1)
            embeddings_last_altered = self.model(last_sub_win_samples_altered).mean(dim=1)

            # Loss for strong sample pairs
            strong_neg_loss = self.compute_negative_loss(embeddings_first, embeddings_last_altered)

            gp = self.gradient_penalty(samples, embeddings)
            # Total Loss with GP
            extended_neg_losses = weak_neg_losses + [strong_neg_loss]  
            total_loss = self.contrastive_loss_function(pos_losses, extended_neg_losses,self.temperature) + self.lamb * gp

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        # Caculate thresold
        threshold_dis = self.calculate_threshold(windows)
        return threshold_dis

    def calculate_threshold(self, windows):
        all_poslosses = []
        for sub_win in windows:
            samples = self.generate_samples(sub_win)
            embeddings = self.model(samples).mean(dim=1)
            _, all_posloss = self.compute_positive_loss(embeddings)
            all_poslosses.append(all_posloss)

        all_losses_tensor = torch.cat(all_poslosses)
        threshold_dis = torch.quantile(all_losses_tensor, self.percentile)
        return threshold_dis

    def test(self, window_data):
        self.model.eval()  # evaluation

        # New sub-window data points at next sliding window
        windows = torch.split(window_data, int(window_data.size(0) / self.sub_window_num))
        embeddings = []
        distances = []

        # h_j and h_j'
        for sub_win in windows:
            sub_win_samples = self.generate_samples(sub_win)
            with torch.no_grad():
                embedding = self.model(sub_win_samples).mean(dim=1)
                embeddings.append(embedding)

        # Distance between the last sub-window and all previous sub-windows in the same sliding window
        last_interval_embedding = embeddings[-1]
        distances = []
        for embedding in embeddings[:-1]:
            distance = self.compute_negative_loss(embedding, last_interval_embedding)
            distances.append(distance)

        return distances