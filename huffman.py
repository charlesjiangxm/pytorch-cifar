import dippykit
import numpy as np
import torch
import sys


def my_huffman_encode(np_array):
    # upload to pytorch gpu for quantization
    original_float = torch.FloatTensor(np_array).cuda()

    # perform quantization on pytorch with gpu
    min_float = torch.min(original_float).item()
    max_float = torch.max(original_float).item()
    scale = (max_float - min_float) / (2 ** 30 - 1)
    zero_point = - 2 ** 30
    quant_float = torch.quantize_per_tensor(original_float, scale, zero_point, torch.qint32)

    # huffman encoding
    quant_int_repr_np = quant_float.int_repr().cpu().numpy()
    im_encoded, stream_length, symbol_code_dict, symbol_prob_dict = dippykit.huffman_encode(quant_int_repr_np)

    return im_encoded, symbol_code_dict


def my_huffman_decode(im_encoded, symbol_code_dict):
    im_decoded = dippykit.huffman_decode(im_encoded, symbol_code_dict)

    return im_decoded


if __name__ == "__main__":
    dummy_original_float = torch.abs(torch.rand(1000)).cuda()
    dummy_dequant_float = dummy_original_float.dequantize()
    print(f"quantization error {torch.norm(dummy_original_float-dummy_dequant_float)}")

    # encode
    dummy_im_encoded, dummy_symbol_code_dict = my_huffman_encode(dummy_original_float)

    # decode
    dummy_im_decoded = my_huffman_decode(dummy_im_encoded, dummy_symbol_code_dict)

    # log
    print(f"original size {dummy_dequant_float.nbytes} Byte")
    print(f"quantized data size {sys.getsizeof(dummy_im_encoded)} Byte")
    print(f"quantized code book size {sys.getsizeof(dummy_symbol_code_dict)} Byte")
