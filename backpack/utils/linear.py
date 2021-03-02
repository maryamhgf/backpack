from torch import einsum
# TODO: remove
from torch import rand
import opt_einsum as oe


def extract_weight_diagonal(module, backproped):
    return einsum("vno,ni->oi", (backproped ** 2, module.input0 ** 2))


def extract_bias_diagonal(module, backproped):
    return einsum("vno->o", backproped ** 2)

# TODO: Add support for NGD here
def extract_weight_ngd(module, backproped):
	#### exact methods ####
	# test: naive method plus
    # A =  einsum("vno,ni->vnoi", (backproped, module.input0))
    # return  einsum("vnoi,kloi->vnkl", (A, A))

    # test: me plus [GOLD]
    B =  einsum("ni,li->nl", (module.input0, module.input0))	
    A =  einsum("vno,klo->vnkl", (backproped, backproped))
    return einsum("vnkl,nl->vnkl", (A, B))

    # test: me plus plus [SILVER]
    # A = einsum("ni,li,vno,klo->vnkl", (module.input0, module.input0, backproped, backproped))
    # return A

    # test: opt_einsum
    # A = oe.contract("ni,li,vno,klo->vnkl", module.input0, module.input0, backproped, backproped)
    # return A

    #### extra approximations ####
    # test: only diagonals:
    # A = einsum("vno,ni->vnoi", (backproped ** 2, module.input0 ** 2))
    # return einsum("vnoi->vn", A)

def extract_bias_ngd(module, backproped):
    return einsum("vno,klo->vnkl", backproped, backproped)
    # return einsum("vno->vn", backproped ** 2)

