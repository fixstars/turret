# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import random
import argparse
import util

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size,
                           device=util.get_device())


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=7):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_output):
        # Embedding and LSTM
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        hidden_repeat = hidden[0].view(-1, self.hidden_size).repeat(
            self.max_length, 1)

        # Get Attention context
        hidden_dot = (encoder_output * hidden_repeat).sum(2)
        hidden_sum = torch.mm(hidden_dot, encoder_output.squeeze(0))
        context = F.softmax(hidden_sum, dim=1).unsqueeze(0)

        # Get output
        output = torch.cat((output, context), 2)
        output = self.out(output).squeeze(0)
        output = F.log_softmax(output, dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size,
                           device=util.get_device())


def _train(idx, x_train, t_train, encoder, decoder, encoder_optimizer,
           decoder_optimizer, criterion, max_length=7):
    encoder_hidden = encoder.initHidden()
    encoder_cell = encoder.initHidden()
    encoder_hiddens = (encoder_hidden, encoder_cell)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = t_train.size(0)

    loss = 0

    encoder_output, encoder_hiddens = encoder(x_train, encoder_hiddens)

    decoder_input = torch.full((1, 1), util.get_id_from_char("_"),
                               dtype=torch.long, device=util.get_device())
    decoder_hiddens = encoder_hiddens

    for di in range(target_length - 1):
        decoder_output, decoder_hiddens = decoder(decoder_input,
                                                  decoder_hiddens,
                                                  encoder_output)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.detach()  # detach from history as input

        loss += criterion(decoder_output, t_train[di + 1])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / (target_length - 1)


def evaluate(encoder, decoder, x, t, char_to_id, id_to_char):
    text_num = len(x)
    target_length = t.shape[1]

    correct_ans = 0
    for i in range(text_num):
        x_batch = x[i]
        t_batch = t[i]

        with torch.no_grad():
            encoder_hidden = encoder.initHidden()
            encoder_cell = encoder.initHidden()
            encoder_hiddens = (encoder_hidden, encoder_cell)

            encoder_output, encoder_hiddens = encoder(x_batch, encoder_hiddens)

            decoder_input = torch.full((1, 1), char_to_id["_"],
                                       dtype=torch.long,
                                       device=util.get_device())
            decoder_hiddens = encoder_hiddens

            decoded_words = []

            for di in range(target_length - 1):
                decoder_output, decoder_hiddens = decoder(decoder_input,
                                                          decoder_hiddens,
                                                          encoder_output)
                topv, topi = decoder_output.data.topk(1, dim=1)
                decoded_words.append(id_to_char[topi.item()])

                decoder_input = topi.squeeze(1).detach()

            a_words = [id_to_char[w.item()] for w in t_batch[1:]]
            a_word = "".join(a_words)
            infer_word = "".join(decoded_words)
            if a_word == infer_word:
                correct_ans += 1

    return (float(correct_ans) / text_num)


def train_all(train_text, valid_text, model_path, n_epoc=10,
              hidden_size=512, learning_rate=0.01):
    x_train, t_train = util.load_text(train_text)
    char_to_id = util.get_dict_char_to_id()
    id_to_char = util.get_dict_id_to_char()
    text_num = len(x_train)
    list_num = list(range(text_num))
    max_dict = util.get_max_dict()
    x_valid, t_valid = util.load_text(valid_text, use_dict=True,
                                      dict_data=char_to_id)

    encoder = Encoder(max_dict, hidden_size).to(util.get_device())
    decoder = Decoder(hidden_size, max_dict, max_length=x_train.shape[1]).to(
                  util.get_device())
    encoder_optimizer = torch.optim.SGD(encoder.parameters(),
                                        lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(),
                                        lr=learning_rate)
    criterion = nn.NLLLoss()
    print_loss_total = 0

    for i in range(1, n_epoc + 1):
        last_print = 0
        shuffle_num = random.sample(list_num, k=text_num)
        x_train = x_train[shuffle_num]
        t_train = t_train[shuffle_num]

        print("epoc {}/{}".format(i, n_epoc), flush=True)
        for j in range(text_num):
            # Flip questions to speed train.
            loss = _train(max_dict, x_train[j].flip(1), t_train[j], encoder,
                          decoder, encoder_optimizer, decoder_optimizer,
                          criterion)
            print_loss_total += loss
            if (j + 1) % 10000 == 0 or j == text_num - 1:
                print_loss_avg = print_loss_total / (j - last_print)
                last_print = j
                print_loss_total = 0
                print("iter {}/{} loss:{}".format(
                      (j + 1), text_num, print_loss_avg),
                      flush=True)
                if j == text_num - 1:
                    accuracy_train = evaluate(encoder, decoder, x_train,
                                              t_train, char_to_id, id_to_char)
                    print("Accuracy(train):{}".format(accuracy_train),
                          flush=True)
                    accuracy_valid = evaluate(encoder, decoder, x_valid,
                                              t_valid, char_to_id, id_to_char)
                    print("Accuracy(valid):{}".format(accuracy_valid),
                          flush=True)

    model = {
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'dict_id_to_char': id_to_char,
        'dict_char_to_id': char_to_id,
        'max_sequence_len': x_train.shape[1]
    }
    torch.save(model, model_path)


def main():
    parser = argparse.ArgumentParser(description="Turret LSTM example(train)")
    parser.add_argument('-t', "--train_text", required=True,
                        help="text for train")
    parser.add_argument('-v', "--validation_text", required=True,
                        help="text for validation")
    parser.add_argument('-m', "--model_path", required=True,
                        help="output model path.")
    parser.add_argument('-e', "--epoc_num", default=10, type=int,
                        help="Epoc number.")
    parser.add_argument('-l', "--learning_rate", default=0.01, type=float,
                        help="Learning rate.")
    parser.add_argument('-i', "--hidden_size", default=512, type=int,
                        help="Hidden size.")

    args = parser.parse_args()
    train_all(args.train_text, args.validation_text, args.model_path,
              n_epoc=args.epoc_num, hidden_size=args.hidden_size,
              learning_rate=args.learning_rate)


if __name__ == "__main__":
        main()
