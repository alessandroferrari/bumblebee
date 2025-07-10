#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
from bumblebee.losses.losses import cross_entropy_loss


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            loss_func,
            num_epochs,
            device,
            eval_freq=20):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._train_loader = train_dataloader
        self._eval_loader = eval_dataloader
        self._loss_func = loss_func
        self._iter_counter = 0
        self._num_epochs = num_epochs
        self._eval_freq = eval_freq
        self._device = device
        self.train_losses = []
        self.eval_losses = []

    def _compute_overall_loss(self, data_loader):
        self._model.eval()
        loss = 0.0
        num_batches = 0
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(self._device)
            target_batch = target_batch.to(self._device)
            with torch.no_grad():
                predicted_logits = self._model.forward(input_batch)
                loss = loss + \
                    self._loss_func(predicted_logits, target_batch, self._device)
            num_batches = num_batches + 1
        loss = loss / num_batches
        return loss

    def evaluate(self):
        train_loss = self._compute_overall_loss(self._train_loader)
        eval_loss = self._compute_overall_loss(self._eval_loader)

        return train_loss, eval_loss

    def train(self):
        num_iter_per_epoch = len(self._train_loader)

        print(f"Running eval for epoch {0}.")
        train_loss, eval_loss = self.evaluate()
        print(f"Train loss: {train_loss} - Eval loss: {eval_loss}.")
        self.train_losses.append(train_loss.to("cpu").item())
        self.eval_losses.append(eval_loss.to("cpu").item())

        for epoch in range(self._num_epochs):

            print(
                f"Running training for epoch {epoch + 1} / {self._num_epochs} .")
            self._model.train()
            for input, target in self._train_loader:
                input = input.to(self._device)
                target = target.to(self._device)
                self._optimizer.zero_grad()
                predicted_logits = self._model.forward(input)
                loss = self._loss_func(predicted_logits, target, self._device)
                print(
                    f"Epoch: {epoch + 1} / {self._num_epochs} - Iter: {self._iter_counter + 1} / {num_iter_per_epoch * self._num_epochs} - Training loss: {loss.item()}.")
                loss.backward()
                self._optimizer.step()
                self._iter_counter = self._iter_counter + 1

            if (self._iter_counter % self._eval_freq) == 0:
                print(f"Running eval for epoch {epoch + 1}.")
                train_loss, eval_loss = self.evaluate()
                print(f"Train loss: {train_loss} - Eval loss: {eval_loss}.")
                self.train_losses.append(train_loss.to("cpu").item())
                self.eval_losses.append(eval_loss.to("cpu").item())

        return self.train_losses, self.eval_losses
