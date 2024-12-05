import torch
from encodingToArch import decode_and_build_unet
import numpy as np

def tester(model_conf, param1xl, param1xu, param2xl, param2xu):
    base_array = np.zeros(12)
    base_array[:4] = 9
    base_array[4::2] = 1
    height = 512
    width = 512
    dim = 64
    batch_size = 12
    channels = 3
    expected_enc0_shape = (batch_size, dim, height, width)
    expected_enc1_shape = (batch_size, dim * 2, height // 2, width // 2)
    expected_enc2_shape = (batch_size, dim * 4, height // 4, width // 4)
    expected_bottle_shape = (batch_size, dim * 8, height // 8, width // 8)
    expected_dec0_shape = (batch_size, dim * 4, height // 4, width // 4)
    expected_dec1_shape = (batch_size, dim * 2, height // 2, width // 2)
    expected_dec2_shape = (batch_size, dim, height, width)
    count = 0
    for i in range(4):
        print(count)
        count += 1
        base_array[i] = model_conf
        for j in range(param1xl, param1xu + 1):
            base_array[i * 2 + 4] = j
            for k in range(param2xl, param2xu + 1):
                base_array[i * 2 + 5] = k
                model = decode_and_build_unet(base_array, dim)
                image = torch.randn(batch_size, channels, height, width)
                print(j, k, i)
                with torch.no_grad():
                    enc0_out, enc1_out, enc2_out, bottle, dec0_out, dec1_out, dec2_out, final = model(image)

                assert enc0_out.shape == expected_enc0_shape, f"enc0_out shape mismatch: {enc0_out.shape}"
                assert enc1_out.shape == expected_enc1_shape, f"enc1_out shape mismatch: {enc1_out.shape}"
                assert enc2_out.shape == expected_enc2_shape, f"enc2_out shape mismatch: {enc2_out.shape}"
                assert bottle.shape == expected_bottle_shape, f"bottle shape mismatch: {bottle.shape}"
                assert dec0_out.shape == expected_dec0_shape, f"dec0_out shape mismatch: {dec0_out.shape}"
                assert dec1_out.shape == expected_dec1_shape, f"dec1_out shape mismatch: {dec1_out.shape}"
                assert dec2_out.shape == expected_dec2_shape, f"dec2_out shape mismatch: {dec2_out.shape}"
                assert final.shape == (batch_size, channels, height, width), f"final image shape mismatch: {final.shape}"

                base_array[i * 2 + 5] = 1

            base_array[i * 2 + 4] = 0

        base_array[i] = 9

tester(1, 1, 8 ,0, 4)