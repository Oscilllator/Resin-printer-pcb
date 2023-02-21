import math
import os

import numpy as np
import matplotlib.pyplot as plt
# See here for how to install on windows:
# https://stackoverflow.com/questions/73637315/oserror-no-library-called-cairo-2-was-found-from-custom-widgets-import-proje
import cairosvg
import imageio

'''
Documentation for the file format comes from:
https://github.com/cbiffle/catibo/blob/master/doc/cbddlp-ctb.adoc#the-layer-table
See there for all the offsets etc.
'''
def decode_image(encoded: np.ndarray, dims: np.ndarray):
    dims = np.flipud(dims)
    assert(encoded.dtype == np.uint8)
    assert(dims.size == 2)
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

# Again, see github link for magic numbers
def parse_cbddlp(buf: np.ndarray):
    layer_table_offset = buf[0x40:0x40+4].view(np.uint32)[0]

    # layer_table = buf[layer_table_offset:layer_table_offset + 0x24]
    # image_offset = layer_table[0xC:0xC+0x4].view(np.uint32)[0]
    # image_len = layer_table[0x10:0x10+0x4].view(np.uint32)[0]
    # # The below dict is a mutable reference to the underlying buffer:
    # out = {
    #     'printer_dims':   buf[8:8+3*4].view(np.float32),
    #     'screen_res':   buf[0x34:0x34+2*4].view(np.uint32),
    #     'layer_table_offset':   buf[0x40:0x40+4].view(np.uint32),
    #     'layer_table':   buf[layer_table_offset:layer_table_offset + 0x24],
    #     'image_offset':   layer_table[0xC:0xC+0x4].view(np.uint32),
    #     'image_len':   layer_table[0x10:0x10+0x4].view(np.uint32),
    #     'z_height': layer_table[0x0:0x0+0x4].view(np.float32),
    #     'exposure_time_s': layer_table[0x4:0x4+0x4].view(np.float32),
    #     'encoded_image':   buf[image_offset:image_offset + image_len],
    # }

    layer_table = buf[layer_table_offset:layer_table_offset + 0x24]
    image_offset = layer_table[0xC:0xC+0x4].view(np.uint32)[0]
    image_len = layer_table[0x10:0x10+0x4].view(np.uint32)[0]
    # The below dict is a mutable reference to the underlying buffer:
    out = {
        'printer_dims':   buf[8:8+3*4].view(np.float32),
        'screen_res':   buf[0x34:0x34+2*4].view(np.uint32),
        'layer_table_offset':   buf[0x40:0x40+4].view(np.uint32),
        'layer_table':   buf[layer_table_offset:layer_table_offset + 0x24],
        'image_offset':   buf[layer_table_offset + 0xC:layer_table_offset+0xC+0x4].view(np.uint32),
        'image_len':   buf[layer_table_offset + 0x10: layer_table_offset + 0x10+0x4].view(np.uint32),
        'z_height': buf[layer_table_offset + 0x0: layer_table_offset + 0x0+0x4].view(np.float32),
        'exposure_time_s': buf[layer_table_offset + 0x4: layer_table_offset + 0x4+0x4].view(np.float32),
        'encoded_image':   buf[image_offset:image_offset + image_len],
    }

    #  make sure it's actually mutable
    for _, v in out.items(): v.flags.writeable = True
    return out

def add_image_to_cbddlp(buf_orig: np.ndarray, image: np.ndarray, exposure_time_s: float = 600):
    image_encoded = encode_image(image.T)
    parsed_orig = parse_cbddlp(buf_orig)
    metadata_sz = buf_orig.size - parsed_orig['image_len'][0]
    print(f"adding image of size {image_encoded.size} to orig size {parsed_orig['image_len']}")
    assert(parsed_orig['image_offset'] + parsed_orig['image_len'] == buf_orig.size)
    assert(metadata_sz > 0)
    out_sz = metadata_sz + image_encoded.size

    buf_out = np.zeros(out_sz, np.uint8)
    buf_out[0:metadata_sz] = buf_orig[0:metadata_sz]

    parsed_out = parse_cbddlp(buf_out)
    # assumption: there is only one layer in the original image
    buf_out[int(parsed_out['image_offset']):] = image_encoded
    # [:] syntax modifies in-place rather than making that variable an int
    parsed_out['image_len'][:] = image_encoded.size
    parsed_out['exposure_time_s'][:] = exposure_time_s
    parsed_out['z_height'][:] = 0  # I hope you took your print head off.
    return buf_out

def trim_whitespace(image: np.ndarray):
    x_extent = np.argwhere(np.sum(image, axis=0)).squeeze()[[0, -1]]
    y_extent = np.argwhere(np.sum(image, axis=1)).squeeze()[[0, -1]]
    image = image[ y_extent[0]:y_extent[1],x_extent[0]:x_extent[1]]
    return image

def parse_gerber_svg(fname: str, dpi: int):
    with open(fname, 'rb') as f:
        buf = f.read()
    png = cairosvg.svg2png(bytestring=buf, dpi=dpi)
    image = imageio.imread(png)
    image = image.sum(axis=-1)  # get rid of rgba channel
    image[image > 0] = 1
    image = image.astype(np.uint8)

    return image

def read_buf(fname: str):
    with open(fname, 'rb') as f:
        buf = f.read()
    buf = np.frombuffer(buf, dtype=np.uint8)
    return buf

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    base_fname = r"single_layer_circle.cbddlp"
    gerber_fname = r"fet_1-F_Cu.svg"

    buf = read_buf(base_fname).copy()
    parsed = parse_cbddlp(buf)
    screen_res = parsed['screen_res']; printer_dims = parsed['printer_dims']

    dpi = 25.4 * screen_res / printer_dims[0:2]
    assert(np.abs(1 - dpi[0] / dpi[1]) < 0.001)  # x and y dpi should be the same
    dpi = int(round(dpi[0]))

    gerber = parse_gerber_svg(gerber_fname, dpi)
    gerber = trim_whitespace(gerber)
    
    # gerber[:] = 1
    if gerber.shape[0] > screen_res[0]:
        gerber = gerber.T
    assert((gerber.shape <= screen_res).all()), "gerber is too big to print"
    out_image = np.zeros(screen_res, dtype=np.uint8)
    out_image[0:gerber.shape[0], 0:gerber.shape[1]] = gerber
    out_image = np.flipud(out_image)
    out_image = 1 - out_image

    buf_out = add_image_to_cbddlp(buf, out_image)
    out_fname = "out.cbddlp"
    with open(out_fname, "wb") as f:
        f.write(buf_out)
    try:
        
        # usb_fname =os.path.join(r"/media/harry/3D PRINTER", usb_fname) 
        usb_fname = os.path.join(r"G:\ "[0:-1], out_fname)
        with open(usb_fname, "wb") as f:
            f.write(buf_out)
        print(f"wrote output to {usb_fname}")
    except (FileNotFoundError, PermissionError): 
        print("could not write to usb")

    if True:
        encoded_image = parsed['encoded_image']
        img = decode_image(encoded_image, screen_res)
        encoded_diy = encode_image(img)
        assert((encoded_image == encoded_diy).all())
        plt.figure(); plt.title("contents of base file")
        plt.imshow(img,  interpolation='none')
        plt.figure(); plt.title("the gerber file to insert")
        plt.imshow(gerber,  interpolation='none')
        plt.figure(); plt.title("the output file with the gerber file inserted")

        output_reloaded = read_buf(out_fname).copy()
        img2 = decode_image(parse_cbddlp(output_reloaded)['encoded_image'], screen_res)
        plt.imshow(img2,  interpolation='none')
        plt.show(); 
