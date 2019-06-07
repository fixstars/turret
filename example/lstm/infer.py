# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import argparse
import util

import numpy as np
import torch
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
from pycuda.compiler import SourceModule # noqa

import turret
import turret.layers as L


def _parse_dtype(s):
    if s == "half":
        return turret.DataType.HALF
    else:
        return turret.DataType.FLOAT


def _load_model(model_path):
    torch_model = torch.load(model_path, map_location="cpu")
    encoder = {}
    for enc_key in torch_model["encoder"].keys():
        enc_value = torch_model["encoder"][enc_key]
        encoder[enc_key] = np.asarray(enc_value)
    decoder = {}
    for dec_key in torch_model["decoder"].keys():
        dec_value = torch_model["decoder"][dec_key]
        decoder[dec_key] = np.asarray(dec_value)
    id_to_char = torch_model["dict_id_to_char"]
    char_to_id = torch_model["dict_char_to_id"]
    max_sequence_len = torch_model["max_sequence_len"]

    return encoder, decoder, id_to_char, char_to_id, max_sequence_len


def _reorg_lstm_parameters(ih, hh):
    hidden_size = len(ih) // 4
    return L.LSTMParameterSet(
        ih[0*hidden_size:1*hidden_size],  # W_input
        hh[0*hidden_size:1*hidden_size],  # U_input
        ih[2*hidden_size:3*hidden_size],  # W_cell
        hh[2*hidden_size:3*hidden_size],  # U_cell
        ih[1*hidden_size:2*hidden_size],  # W_forget
        hh[1*hidden_size:2*hidden_size],  # U_forget
        ih[3*hidden_size:4*hidden_size],  # W_output
        hh[3*hidden_size:4*hidden_size])  # U_output


def build_encodeengine(encoder, batch_size, dtype, logger,
                       max_sequence_length=16, workspace_size=2**30):
    sys.stderr.write("------------------------------\n")
    sys.stderr.write(" encoder\n")
    sys.stderr.write("------------------------------\n")
    builder = turret.InferenceEngineBuilder(logger)
    network = builder.create_network(dtype)

    # extract parameters
    emb = encoder["embedding.weight"]
    weights = []
    bias = []
    weights_rev = []
    bias_rev = []
    bidirect = ("lstm.weight_hh_l0_reverse" in encoder)
    weights.append(_reorg_lstm_parameters(
        encoder["lstm.weight_ih_l0"][:],
        encoder["lstm.weight_hh_l0"][:]))
    bias.append(_reorg_lstm_parameters(
        encoder["lstm.bias_ih_l0"][:],
        encoder["lstm.bias_hh_l0"][:]))
    if bidirect:
        weights_rev.append(_reorg_lstm_parameters(
            encoder["lstm.weight_ih_l0_reverse"][:],
            encoder["lstm.weight_hh_l0_reverse"][:]))
        bias_rev.append(_reorg_lstm_parameters(
            encoder["lstm.bias_ih_l0_reverse"][:],
            encoder["lstm.bias_hh_l0_reverse"][:]))

    # define a network
    src = network.add_constant(emb)
    h = network.add_input(
        "words", turret.DataType.INT32,
        turret.Dimensions((
            (1, turret.DimensionType.INDEX),
            (max_sequence_length, turret.DimensionType.INDEX))))
    h = L.gather(src, h, 0)

    h_lengths = network.add_input(
        "lengths", turret.DataType.INT32,
        turret.Dimensions(((1, turret.DimensionType.INDEX),)))

    if bidirect:
        context, hidden, cell = L.blstm_v2(
            h, max_sequence_length, weights, weights_rev, bias, bias_rev,
            sequence_lengths=h_lengths)
    else:
        context, hidden, cell = L.lstm_v2(
            h, max_sequence_length, weights, bias,
            sequence_lengths=h_lengths)

    network.mark_output("context", context)
    network.mark_output("hidden", hidden)
    network.mark_output("cell", cell)

    builder.max_batch_size = batch_size
    builder.max_workspace_size = workspace_size

    # build
    engine = builder.build(network)
    return engine


def build_decodeengine(decoder, batch_size, dtype, logger,
                       max_sequence_length=16, workspace_size=2**30):
    sys.stderr.write("------------------------------\n")
    sys.stderr.write(" decoder\n")
    sys.stderr.write("------------------------------\n")
    builder = turret.InferenceEngineBuilder(logger)
    network = builder.create_network(dtype)

    emb = decoder["embedding.weight"]
    hidden_size = decoder["lstm.weight_hh_l0"].shape[1]

    weights = []
    bias = []
    weights.append(_reorg_lstm_parameters(decoder["lstm.weight_ih_l0"],
                                          decoder["lstm.weight_hh_l0"]))
    bias.append(_reorg_lstm_parameters(decoder["lstm.bias_ih_l0"],
                                       decoder["lstm.bias_hh_l0"]))

    tgt = network.add_constant(emb)

    # Embedding and LSTM.
    h_indices_in = network.add_input(
        "indices_in", turret.DataType.INT32,
        turret.Dimensions(((1, turret.DimensionType.INDEX),)))
    h_indices_in = L.gather(tgt, h_indices_in, 0)
    h_indices_in = L.reshape(h_indices_in,
                             turret.Dimensions.CHW(1, 1, hidden_size))
    h_hidden = network.add_input(
        "hidden_in", turret.DataType.FLOAT,
        turret.Dimensions.CHW(1, 1, hidden_size))
    h_cell = network.add_input(
            "cell_in", turret.DataType.FLOAT,
            turret.Dimensions.CHW(1, 1, hidden_size))
    h, h_hidden, h_cell = L.lstm_v2(h_indices_in, 1, weights, bias,
                                    hidden_state=h_hidden, cell_state=h_cell)
    network.mark_output("hidden_out", h_hidden)
    network.mark_output("cell_out", h_cell)

    # Attention.
    h_hidden_enc = network.add_input(
            "enc_hidden", turret.DataType.FLOAT,
            turret.Dimensions.CHW(1, max_sequence_length, hidden_size))
    h_attn_w = L.elementwise(h_hidden_enc, h_hidden,
                             turret.ElementWiseOperation.PROD)
    h_attn_w = L.reduce(h_attn_w, turret.ReduceOperation.SUM, axes=2)
    h_hidden_enc = L.reshape(h_hidden_enc,
                             turret.Dimensions.HW(max_sequence_length,
                                                  hidden_size))
    h_context = L.matrix_multiply(h_attn_w, False, h_hidden_enc, False)
    h_context = L.softmax(h_context)
    h_context = L.reshape(h_context,
                          turret.Dimensions.CHW(
                            1, h_context.dimensions.shape[0],
                            h_context.dimensions.shape[1]))
    h = L.concat([h, h_context], axis=2)

    # Out, softmax, and log.
    out_weights = decoder["out.weight"][:]
    out_bias = decoder["out.bias"][:]
    h = L.fully_connected(h, out_weights, out_bias)
    h = L.softmax(h)
    h = L.unary(h, turret.UnaryOperation.LOG)
    h = L.reshape(h, turret.Dimensions(((h.dimensions.shape[0],
                                        turret.DimensionType.SPATIAL),)))
    _, h_indices_out = L.top_k(h, turret.TopKOperation.MAX, 1, 1)
    h_indices_out.dimensions  # If this line is removed, error is occurred.
    network.mark_output("indices_out", h_indices_out)

    builder.max_batch_size = batch_size
    builder.max_workspace_size = workspace_size

    # build
    engine = builder.build(network)
    return engine


def inference(infer_text, batch_size, model_path, dtype):
    # Load model and text
    encoder, decoder, id_to_char, char_to_id, max_sequence_len = \
        _load_model(model_path)
    dtype_trt = _parse_dtype(dtype)

    x, t = util.load_text(infer_text, use_dict=True, dict_data=char_to_id)
    text_num = len(x)

    # Build engine
    logger = turret.loggers.ConsoleLogger()
    encoder_engine = build_encodeengine(encoder, batch_size, dtype_trt,
                                        logger, max_sequence_len)
    decoder_engine = build_decodeengine(decoder, batch_size, dtype_trt,
                                        logger, max_sequence_len)

    # Create execution contexts
    encoder_trt = turret.ExecutionContext(encoder_engine)
    encbuf = encoder_trt.create_buffer()
    decoder_trt = turret.ExecutionContext(decoder_engine)
    decbuf = decoder_trt.create_buffer()
    stream = cuda.Stream()

    # Inference
    inference_num = 0
    correct_ans = 0
    for head in range(0, text_num, batch_size):
        tail = min(head + batch_size, text_num)
        real_batch_size = tail - head

        x_batch = x[head:tail]
        t_batch = t[head:tail]

        # Encoder
        src_words = np.full((real_batch_size, 1, max_sequence_len),
                            1, dtype=np.int32)
        src_lengths = np.empty((real_batch_size, 1), dtype=np.int32)
        for i, words in enumerate(x_batch):
            length = len(words)
            src_words[i, 0, :length] = [w.item() for w in words]
            src_lengths[i, 0] = length
        encbuf.put("words", src_words)
        encbuf.put("lengths", src_lengths)
        encoder_trt.enqueue(encbuf, stream)
        stream.synchronize()

        # Decoder
        enc_context = encbuf.get("context")
        decbuf.put("enc_hidden", enc_context)
        decbuf.put("hidden_in", encbuf.get("hidden"))
        decbuf.put("cell_in", encbuf.get("cell"))
        indices = np.full((real_batch_size, 1), char_to_id["_"],
                          dtype=np.int32)
        decbuf.put("indices_in", indices)

        all_words = []
        for i in range(t.shape[1] - 1):
            decoder_trt.enqueue(decbuf, stream)
            stream.synchronize()

            indices_out = decbuf.get("indices_out")
            indices_out = indices_out.reshape((indices_out.shape[0],)).tolist()
            words = [[id_to_char[idx]] for idx in indices_out]
            if i == 0:
                all_words = words
            else:
                for j in range(real_batch_size):
                    all_words[j].append(words[j][0])

            decbuf.swap("hidden_in", "hidden_out")
            decbuf.swap("cell_in", "cell_out")
            decbuf.swap("indices_in", "indices_out")

        # Print result
        for i in range(real_batch_size):
            q_words = [id_to_char[w.item()] for w in x_batch[i]]
            a_words = [id_to_char[w.item()] for w in t_batch[i]]
            q_word = "".join(q_words)
            a_word = "".join(a_words).split('_')[1]
            infer_word = "".join(all_words[i])
            print("Question{}: {}".format(inference_num + 1, q_word))
            print("Inference{}: {}".format(inference_num + 1, infer_word))
            print("Answer{}: {}".format(inference_num + 1, a_word))
            if infer_word == a_word:
                print("Correct.")
                correct_ans += 1
            else:
                print("Incorrect.")
            print("\n")
            inference_num += 1

    print("Accuracy:{}".format(float(correct_ans) / inference_num))

    return


def main():
    parser = argparse.ArgumentParser(
                 description="Turret LSTM example(inference)")
    parser.add_argument('-t', "--infer_text", required=True,
                        help="text for inference")
    parser.add_argument('-m', "--model_path", required=True,
                        help="model path.")
    parser.add_argument('-b', "--batch_size", default=20, type=int,
                        help="batch size")
    parser.add_argument("-d", "--dtype", default="float",
                        choices=["float", "half"],
                        help="data type for computations in inference")

    args = parser.parse_args()
    inference(args.infer_text, args.batch_size, args.model_path, args.dtype)


if __name__ == "__main__":
        main()
