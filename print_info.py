import math

import numpy as np
import matplotlib.pyplot as plt

'''
Documentation for the file format comes from:
https://github.com/cbiffle/catibo/blob/master/doc/cbddlp-ctb.adoc#the-layer-table
See there for all the offsets etc.
'''
def decode_image(encoded: np.ndarray, dims: np.ndarray):
    assert(encoded.dtype == np.uint8)
    assert(dims.size == 2)
    dims = np.flipud(dims)
    values = ((encoded & (1 << 7)) > 0).astype(np.int8)
    lengths = (encoded & 0b01111111).astype('int')
    ptr = 0
    decoded = np.empty(dims, dtype=np.int8)
    for i in range(encoded.size):
        start = ptr 
        len = lengths[i]
        decoded.reshape((-1))[start:start+len] = values[i]
        ptr += len
    return decoded

def encode_image(decoded: np.ndarray):
    pixels = decoded.flatten()
    change_locs = np.argwhere(np.diff(pixels, prepend=1337)).squeeze()
    assert(change_locs[0] == 0)
    run_lengths = np.diff(change_locs, append=pixels.size)
    assert(np.sum(run_lengths) == pixels.size)
    values = pixels[change_locs]

    kMaxRunLen = 125
    encoded = []
    for i in range(change_locs.size):
        value = values[i]
        total_run_length = run_lengths[i]
        run_length = 0 
        for _ in range(math.ceil(total_run_length / kMaxRunLen)):
            this_run_length = min(total_run_length - run_length, kMaxRunLen)
            assert(this_run_length <= kMaxRunLen)
            encoded.append((value << 7) | this_run_length)
            run_length += this_run_length
    encoded = np.array(encoded, dtype=np.uint8)
    return encoded


fname = r"C:\Users\harry\Documents\Git\catibo\_single_layer_circle.cbddlp"

with open(fname, 'rb') as f:
    buf = f.read()
buf = np.frombuffer(buf, dtype=np.uint8)

printer_dims = buf[8:8+3*4].view(np.float32)
screen_res = buf[0x34:0x34+2*4].view(np.uint32)
print(f"{printer_dims=}, {screen_res=}")
layer_table_offset = buf[0x40:0x40+4].view(np.uint32)[0]
print(f"layer offset: {hex(layer_table_offset)}, {layer_table_offset}")
layer_table = buf[layer_table_offset:layer_table_offset + 0x24]
print(layer_table)
exposure = layer_table[0x4:0x4+0x4].view(np.float32)
image_offset = layer_table[0xC:0xC+0x4].view(np.uint32)[0]
image_len = layer_table[0x10:0x10+0x4].view(np.uint32)[0]
encoded_image = buf[image_offset:image_offset + image_len]
print(encoded_image)

img = decode_image(encoded_image, screen_res)
encoded_diy = encode_image(img)
assert((encoded_image == encoded_diy).all())
plt.imshow(img,  interpolation='none')
plt.show()

