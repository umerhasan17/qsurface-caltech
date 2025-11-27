import numpy as np
from qldpc import abstract, codes


def create_lifted_product_code():
    bits_checks_a: tuple[int, int] = (5, 3)
    bits_checks_b: tuple[int, int] = (3, 5)
    field: int = 2
    seed: int = 42

    code_a = codes.ClassicalCode.random(*bits_checks_a, field=field, seed=seed)

    # code_b = codes.ClassicalCode.random(*bits_checks_b, field=field, seed=seed)

    g = abstract.CyclicGroup(31)
    matrix_a = abstract.RingArray.build(code_a.matrix, g)
    matrix_b = abstract.RingArray.build(code_a.matrix.T, g)
    code_LP = codes.LPCode(matrix_a, matrix_b)

    print(code_LP.get_code_params())

    return code_LP.matrix_z, code_LP.matrix_x


if __name__ == '__main__':
    print(create_lifted_product_code())
