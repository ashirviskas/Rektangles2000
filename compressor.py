import sys
import numpy as np
import NetworkLoader as nl
import glob
import ImageReader as ir
import cv2


def init_network(model_path):
    network_loader = nl.NetworkLoader(model_path)
    return network_loader


def compress_images(network_loader, take_imagepath, put_imagepath):
    encoder_network = network_loader.get_encoder()
    images = np.array(ir.read_file(take_imagepath)) / 255
    images = np.expand_dims(images, 0)
    encoded_images = encoder_network.predict(images)
    encoded_images = np.array(encoded_images, dtype=np.bool)
    encoded_images = np.packbits(encoded_images, axis=1)
    np.save(put_imagepath, encoded_images)
    # with open(put_imagepath, 'wb') as datafile:
    #     datafile.write(encoded_images)


def decompress_images(network_loader, take_imagepath, put_imagepath):
    decoder_network = network_loader.get_decoder()
    compressed_data = np.load(take_imagepath, mmap_mode="r+")
    compressed_data = np.unpackbits(compressed_data, axis=1)
    print(compressed_data.shape)
    image = decoder_network.predict(compressed_data)
    Z = 255 * image[0] / image[0].max()
    Z = np.array(Z, dtype=np.uint8)
    cv2.imwrite(put_imagepath, Z)
    pass


def parse_args():
    model_path = "./64i_15k_300_retrained_HSV_200more"
    compress = False
    if "--model" in sys.argv:
        model_path = sys.argv[sys.argv.index("--model") + 1]
    if "--c" in sys.argv:
        cur_index = sys.argv.index("--c")
        compress = True
    elif "--d" in sys.argv:
        cur_index = sys.argv.index("--d")
    else:
        print("Bad syntax")
        return
    take_imagepath = sys.argv[cur_index + 1]
    put_imagepath = sys.argv[cur_index + 2]
    return model_path, compress, take_imagepath, put_imagepath


def main():
    model_path, compress, take_imagepath, put_imagepath = parse_args()
    network_loader = init_network(model_path)
    if compress:
        compress_images(network_loader, take_imagepath, put_imagepath)
    else:
        decompress_images(network_loader, take_imagepath, put_imagepath)


if __name__ == '__main__':
    main()
