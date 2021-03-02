from backpack.extensions.mat_to_mat_jac_base import MatToJacMat


class TRIALBaseModule(MatToJacMat):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives, params=params)

    # TODO: Saeed, add backprop for NGD
    # it should be normal backprop to get jacobians
    # maybe even no change


