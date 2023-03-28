#  extract the diagnostic networks

import os
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == '__main__':
    # load the trained IntensityDiagnosticModule:I
    I_DiagnosticModule = load_model('./model_save/ImgModel_T2.hdf5', compile=False)
    # I_DiagnosticModule.summary()
    # extract the intensity diagnostic network
    I_DiagnosticNetwork = I_DiagnosticModule.layers[1].layer
    I_DiagnosticNetwork._name = 'I_DiagnosticNetwork'
    I_DiagnosticNetwork.summary()
    I_DiagnosticNetwork.save('./model_save/I_DiagnosticNetwork.hdf5')

    # load the trained IntegratedDiagnosticModule: Intensity+Size = IS
    I_S_IntegratedDiagnosticModule = load_model('./model_save/FusionModel_T2.hdf5', compile=False)
    # I_S_IntegratedDiagnosticModule.summary()
    # extract the integrated diagnostic network
    I_S_DiagnosticNetwork = I_S_IntegratedDiagnosticModule.layers[1].layer
    I_S_DiagnosticNetwork._name = 'I_S_DiagnosticNetwork'
    I_S_DiagnosticNetwork.summary()
    I_S_DiagnosticNetwork.save('./model_save/I_S_DiagnosticNetwork.hdf5')

    # load the trained IntegratedDiagnosticModule: Intensity+Size+ADC = ISA
    I_S_A_IntegratedDiagnosticModule = load_model('./model_save/FusionModel_T2_ADC.hdf5', compile=False)
    # I_S_A_IntegratedDiagnosticModule.summary()
    # extract the integrated diagnostic network
    I_S_A_DiagnosticNetwork = I_S_A_IntegratedDiagnosticModule.layers[1].layer
    I_S_A_DiagnosticNetwork._name = 'I_S_A_DiagnosticNetwork'
    I_S_A_DiagnosticNetwork.summary()
    I_S_A_DiagnosticNetwork.save('./model_save/I_S_A_DiagnosticNetwork.hdf5')

