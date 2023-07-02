from mclbn256 import Fr
import json


def deserialize( # deserialize the output bytes
    output_bytes,
):
    reconstructed_bytes=bytes()

    for i in output_bytes:
        reconstructed_bytes += i.to_bytes(8, byteorder='little')
    
    rep = Fr.deserialize(reconstructed_bytes)

    return rep.__int__()


    
def get_scaled_value ( # get the scaled value of the output bytes
    witness_path=None, # path to witness json file
    settings_path=None, # path to settings json file
    output_bytes=None, # output bytes (used if witness_path is None)
    scale=7, # scale factor used in gen_witness (used if settings_path is None)
) :
    output_bytes_to_use = get_x_or_use_y (witness_path, 'output_data', output_bytes, "No output bytes provided")
    scale_to_use = get_x_or_use_y (settings_path, 'scale', scale, "No scale provided")
    
    res = []
    for i in output_bytes_to_use:
        res.append(deserialize(i) / 2 ** scale_to_use)

    return res



def calculate_error(
    output_bytes,
    original_output_value,
    scale,
) : # calculate the error with the given output bytes, original output value, and scale
    rep = deserialize(output_bytes)
    
    unscaled_rep = rep / 2 ** scale
    
    return (unscaled_rep) - original_output_value


def calculate_error_of_witness( # calculate the error of the witness (loads from json files)
        witness_path=None, # path to witness json file
        origin_data_path=None, # path to original data json file
        settings_path=None, # path to settings json file
        scale=7, # scale factor used in gen_witness (used if settings_path is None)
        original_output_value=None, # original output value (used if origin_data_path is None)
        output_bytes=None, # output bytes (used if witness_path is None)
):
    output_bytes_to_use = get_output_bytes_to_use(witness_path, output_bytes)
    original_output_value_to_use = get_original_output_value_to_use(origin_data_path, original_output_value)
    scale_to_use = get_scale_to_use(settings_path, scale)
    
    if len(output_bytes_to_use) != len(original_output_value_to_use):
        raise Exception("Output bytes and original output value must be the same length")

    result = []
    for i in range(len(output_bytes_to_use)):
        result.append(calculate_error(
            output_bytes_to_use[i],
            original_output_value_to_use[i],
            scale_to_use,
        ))    

    return result



def get_x_or_use_y (x_path, item_in_json,y, error_message):
    if x_path is not None:
        return json.load(open(x_path, 'r'))[item_in_json] 
    elif y is not None:
        return y
    else:
        raise Exception(error_message)


def get_output_bytes_to_use (witness_path,output_bytes ):
    return get_x_or_use_y (witness_path, 'output_data', output_bytes, "No output bytes provided")


def get_original_output_value_to_use (origin_data_path, original_output_value):
    return get_x_or_use_y (origin_data_path, 'output_data', original_output_value, "No original output value provided")


def get_scale_to_use (settings_path, scale):
    return get_x_or_use_y (settings_path, 'scale', scale, "No scale provided")


if __name__ == '__main__':
    # this represents the 4 numbers in the array of ints you get back
    first_byte = 126
    second_byte = 0
    third_byte = 0
    fourth_byte = 0

    print(calculate_error_of_witness(output_bytes=[[first_byte, second_byte, third_byte, fourth_byte]],original_output_value= [0.49169921875], scale=8))