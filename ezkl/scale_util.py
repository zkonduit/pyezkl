from mclbn256 import Fr
import json
from enum import Enum


class ErrorCalculationType(Enum):
    ADDITIVE = 1
    MULTIPLICATIVE = 2


def deserialize(
    output_bytes,
):
    """deserializes the output bytes and returns the deserialized output
    Arguments:
        output_bytes: e.g. [126,0,0,0]
        - output bytes that are already serialized
    Returns:
        rep: e.g. 126
        - the deserialized output
    """
    reconstructed_bytes = bytes()

    for i in output_bytes:
        reconstructed_bytes += i.to_bytes(8, byteorder="little")

    rep = Fr.deserialize(reconstructed_bytes)

    return rep.__int__()


def get_scaled_value(witness_path, settings_path):
    """calculates the scaled value of the output bytes in the witness
    Arguments:
        witness_path: e.g. "witness.json"
        - path to witness json file
        settings_path: e.g. "settings.json"
        - path to settings json file to get the scale factor
    Returns:
        res: e.g. [126]
        - the scaled value of the output bytes
    """

    output_bytes_to_use = []
    with open(witness_path, "r") as f:
        output_bytes_to_use = json.load(f)["output"]

    scale_to_use = 0
    with open(settings_path, "r") as f:
        scale_to_use = json.load(f)["run_args"]["scale"]

    res = []
    for i in output_bytes_to_use:
        res.append(deserialize(i) / 2**scale_to_use)

    return res


def calculate_error(
    output_bytes,
    original_output_value,
    scale,
    error_calculation_type=ErrorCalculationType.MULTIPLICATIVE,
):
    """calculates the error with the given output bytes, original output value,
      and scale
    Arguments:
        output_bytes: e.g. [126,0,0,0]
        - output bytes that are already serialized
        original_output_value: e.g. 126
        - the original output value
        scale: e.g. 7
        - the scale factor used in gen_witness
        error_calculation_type: e.g. ErrorCalculationType.MULTIPLICATIVE
        - the type of error calculation to use
        Note: if error_calculation_type is ErrorCalculationType.MULTIPLICATIVE, it calculates the percentage of error. 
        Note: If error_calculation_type is ErrorCalculationType.ADDITIVE, it calculates the difference between the original output value and the scaled output value.
    Returns:
        (scaled_rep) - original_output_value: e.g. 0
        - the error of the output bytes
    """
    rep = deserialize(output_bytes)

    scaled_rep = rep / 2**scale
    print(scaled_rep, original_output_value)
    if error_calculation_type == ErrorCalculationType.MULTIPLICATIVE:
        return (
            ((scaled_rep - original_output_value) / original_output_value)
        )
    return (scaled_rep) - original_output_value


def calculate_error_of_witness( 
    witness_path, 
    origin_data_path, 
    settings_path,
    error_calculation_type=ErrorCalculationType.MULTIPLICATIVE,
):
    """calculates the error of the witness
    Arguments:
        witness_path: e.g. "witness.json"
        - path to witness json file
        origin_data_path: e.g. "input_original.json"
        - path to original input json file
        settings_path: e.g. "settings.json"
        - path to settings json file to get the scale factor
        error_calculation_type: e.g. ErrorCalculationType.MULTIPLICATIVE
        - the type of error calculation to use
        Note: if error_calculation_type is ErrorCalculationType.MULTIPLICATIVE, it calculates the percentage of error.
        Note: If error_calculation_type is ErrorCalculationType.ADDITIVE, it calculates the difference between the original output value and the scaled output value.
    Returns:
        result: e.g. [[0.1]]
        - the error of the witness
    Raises:
        IndexError: if the output bytes and original output value are not the same length
    """

    original_output_value = get_original_output_value(origin_data_path)
    output_bytes = get_output_bytes(witness_path)
    scale = get_scale(settings_path)
    
    if len(output_bytes) != len(original_output_value):
        raise IndexError(
            "Output bytes and original output value must be the same length"
        )

    result = []
    for i in range(len(output_bytes)):
        result.append([])
        if len(output_bytes[i]) != len(original_output_value[i]):
            raise IndexError(
                "Output bytes and original output value must be the same length"
            )
        for j in range(len(output_bytes[i])):
            result[i].append(
                calculate_error(
                    output_bytes[i][j],
                    original_output_value[j][i],
                    scale,
                    error_calculation_type
                )
            )

    return result


def get_output_bytes(witness_path):
    with open(witness_path, "r") as f:
        return json.load(f)["output_data"]


def get_original_output_value(origin_data_path):
    with open(origin_data_path, "r") as f:
        return json.load(f)["output_data"]


def get_scale(settings_path):
    with open(settings_path, "r") as f:
        return json.load(f)["run_args"]["scale"]


if __name__ == "__main__":
    # this represents the 4 numbers in the array of ints you get back
    first_byte = 126
    second_byte = 0
    third_byte = 0
    fourth_byte = 0

    print(
        calculate_error_of_witness(
            witness_path="test/test_scale/input_scaled.json",
            origin_data_path="test/test_scale/input_original.json",
            settings_path="test/test_scale/setting.json",
        )
    )
