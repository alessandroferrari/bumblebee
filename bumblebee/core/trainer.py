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
            num_epochs,
            device,
            eval_freq=20):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._train_loader = train_dataloader
        self._eval_loader = eval_dataloader
        self._iter_counter = 0
        self._num_epochs = num_epochs
        self._eval_freq = eval_freq
        self._device = device
        self.train_losses = []
        self.eval_losses = []

    def evaluate(self):
        self._model.eval()

        train_loss = 0.0
        num_batches_training = 0
        for input, target in self._train_loader:
            input = input.to(self._device)
            target = target.to(self._device)
            num_batches_training = num_batches_training + 1
            predicted_logits = self._model.forward(input.to(self._device))
            train_loss = train_loss + \
                cross_entropy_loss(predicted_logits, target.to(self._device))
        train_loss = train_loss / num_batches_training

        eval_loss = 0.0
        num_batches_eval = 0
        for input, target in self._eval_loader:
            input = input.to(self._device)
            target = target.to(self._device)
            predicted_logits = self._model.forward(input.to(self._device))
            eval_loss = eval_loss + \
                cross_entropy_loss(predicted_logits, target.to(self._device))
            num_batches_eval = num_batches_eval + 1
        eval_loss = eval_loss / num_batches_eval

        return train_loss, eval_loss

    def train(self):
        for epoch in range(self._num_epochs):
            print(f"Running training for epoch {epoch}.")
            self._model.train()
            for input, target in self._train_loader:
                input = input.to(self._device)
                target = target.to(self._device)
                self._optimizer.zero_grad()
                predicted_logits = self._model.forward(input)
                loss = cross_entropy_loss(predicted_logits, target)
                print(f"Training loss: {loss.item()}.")
                loss.backward()
                self._optimizer.step()
                self._iter_counter = self._iter_counter + 1

            if (self._iter_counter % self._eval_freq) == 0:
                print(f"Running eval for epoch {self._iter_counter}.")
                train_loss, eval_loss = self.evaluate()
                print(f"Train loss: {train_loss} - Eval loss: {eval_loss}.")
                self.train_losses.append(train_loss.to("cpu").item())
                self.eval_losses.append(eval_loss.to("cpu").item())

        return self.train_losses, self.eval_losses
