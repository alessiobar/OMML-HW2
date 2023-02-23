
#Importing the necessary libraries

import pandas as pd, numpy as np
from scipy.optimize import minimize
import scipy.optimize
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

'''
#############################################
		Dataset
#############################################
'''
df = pd.DataFrame(np.array([[-1.68156303e-02,  1.49361613e+00, -9.96793759e-01],
       [-9.54450459e-02,  2.74278733e+00, -1.16915490e+00],
       [-1.90014212e+00,  6.74755716e-01,  2.56156847e-01],
       [-9.82228845e-01,  1.46448304e+00,  8.80567438e-01],
       [ 9.19084459e-01,  1.49532170e+00, -8.63522815e-01],
       [-1.06740703e+00,  1.58266813e+00,  1.46630109e+00],
       [-1.98190522e+00, -7.27572190e-01, -1.64370083e+00],
       [ 6.07004659e-01,  1.75515471e+00, -9.16813026e-01],
       [ 1.27023399e+00,  5.93176713e-01, -8.46374447e-01],
       [-1.32424366e+00,  2.01166491e+00,  3.33286164e+00],
       [ 1.24797982e+00,  7.97725193e-02, -7.49105582e-01],
       [ 8.81936833e-01,  1.17980193e+00, -9.40206993e-01],
       [ 9.21811080e-01,  2.19912631e+00, -1.05706402e+00],
       [-1.96801001e+00,  1.86164370e+00,  4.29320588e+00],
       [-1.62786016e+00,  1.08877437e+00,  1.81597481e+00],
       [-1.53622509e+00,  7.61140748e-01,  4.14221719e-01],
       [-7.00634706e-01,  2.16646540e+00,  4.26592132e-01],
       [-4.05942800e-01, -1.76274110e+00, -3.75900543e-01],
       [-4.85903347e-01, -1.19877494e+00, -9.01269074e-01],
       [-1.01563690e-01, -2.53595702e+00, -1.52700539e-01],
       [-1.92522560e+00,  1.23841609e+00,  2.59430982e+00],
       [-4.67857624e-01,  6.73234125e-02, -2.89733772e+00],
       [ 1.18221774e-01, -1.19074109e+00, -7.26252644e-01],
       [-1.59436926e+00, -2.27469726e+00, -1.14146483e+00],
       [-1.25412129e+00,  1.17475491e+00,  1.14805912e+00],
       [ 1.90437420e+00, -1.81018790e+00,  3.30583096e+00],
       [-2.09456241e-01,  2.31986077e+00, -7.96093818e-01],
       [ 1.40727829e+00,  1.24927346e+00, -8.03548631e-01],
       [-1.06215886e+00, -1.52718060e+00, -7.79983507e-01],
       [-1.18198589e+00, -2.45566873e+00, -7.83566930e-01],
       [ 4.62883214e-01,  8.82908650e-01, -1.60371191e+00],
       [-2.98170061e-01, -2.91216566e+00, -2.73059645e-01],
       [-1.18075375e-01, -1.52004925e+00, -4.11639457e-01],
       [-8.69902888e-01,  4.58454498e-01, -1.62073573e+00],
       [-1.24305407e+00,  9.41413906e-01,  3.94782966e-01],
       [-4.87644386e-01,  1.01099007e+00, -1.27059599e+00],
       [ 1.75961197e+00,  1.22852403e-01, -3.47913922e-01],
       [ 1.94072882e+00,  2.42165816e+00, -1.46522185e+00],
       [ 2.30626541e-01,  1.08261849e+00, -1.45685498e+00],
       [-1.34280284e+00,  5.72017596e-01, -4.47406170e-01],
       [ 1.01562810e+00,  1.86383126e+00, -9.30341719e-01],
       [-1.50507580e+00, -2.73311749e+00, -1.08418270e+00],
       [-3.88161037e-02,  2.04738757e+00, -8.71027969e-01],
       [-3.17923780e-01,  1.57443989e+00, -7.19081865e-01],
       [ 5.89415726e-01,  1.17546911e+00, -1.12195901e+00],
       [-1.25280447e+00,  2.50811163e+00,  2.62461876e+00],
       [-5.15641356e-01, -2.84658597e+00, -3.71476247e-01],
       [-1.23358567e+00,  2.05984111e+00,  2.92082102e+00],
       [ 6.06014377e-01, -1.46935721e+00,  3.14306228e-01],
       [ 1.61774151e+00,  7.71090477e-01, -7.17285845e-01],
       [-1.80712693e+00,  2.36746829e+00,  4.24108577e+00],
       [ 5.19056923e-02, -2.39416425e-01, -3.08017274e+00],
       [ 1.78677892e+00, -5.57928089e-01,  1.12914212e+00],
       [ 1.65259668e+00,  1.34241847e+00, -8.68200075e-01],
       [-1.56295376e-01, -1.75194468e+00, -2.67896936e-01],
       [ 1.48359307e+00,  2.86681788e+00, -1.52292370e+00],
       [-1.58477600e+00,  1.34216769e+00,  2.72442061e+00],
       [-6.87857448e-01,  1.09876701e+00, -7.19759813e-01],
       [ 1.36091242e+00, -9.53476739e-01,  1.63701035e+00],
       [-1.96686595e-01,  8.38535918e-01, -1.95564773e+00],
       [-1.59193910e+00,  2.17307055e+00,  4.34149064e+00],
       [-9.77217219e-01,  2.82760288e-01, -1.64342162e+00],
       [ 1.26567141e-01,  1.21453361e-01, -3.23240832e+00],
       [-1.22248490e+00, -1.96482846e-01, -1.51138517e+00],
       [ 1.49843991e+00, -7.41857208e-01,  1.35286135e+00],
       [-5.22899996e-01,  2.97771324e+00, -9.07656546e-01],
       [ 1.60512176e+00,  4.96025348e-01, -6.40389069e-01],
       [ 1.92311300e+00, -2.06908365e+00,  2.57580749e+00],
       [-1.19486970e+00,  1.35811908e+00,  1.46884317e+00],
       [ 4.40348665e-01,  1.52123554e+00, -9.68325787e-01],
       [-1.87719510e+00,  3.24666746e-01, -7.89609687e-01],
       [ 1.02890996e+00, -1.51602343e+00,  1.42175698e+00],
       [-2.49212669e-01, -1.08400073e+00, -1.09747018e+00],
       [ 1.05457640e+00, -1.61356479e+00,  1.51262250e+00],
       [-1.86738923e+00,  8.80661784e-01,  1.09134340e+00],
       [ 1.77374312e+00, -2.27339268e+00,  1.85455437e+00],
       [ 8.81616412e-02,  2.70736520e+00, -1.23760730e+00],
       [-7.61274705e-01, -1.25653012e+00, -8.41814073e-01],
       [ 4.48688938e-02,  2.62498495e+00, -1.16852277e+00],
       [ 8.64679475e-03, -7.00536225e-01, -2.00174066e+00],
       [ 1.18795038e+00, -1.11334166e+00,  1.48147256e+00],
       [ 1.39953505e+00, -6.59540845e-01,  9.05201100e-01],
       [ 1.88848659e+00,  2.62636714e+00, -1.54951218e+00],
       [ 1.78529537e+00, -1.63108795e+00,  3.50529725e+00],
       [-1.44774746e+00, -1.81004264e+00, -1.01133268e+00],
       [-6.23928230e-01,  1.23954689e+00, -6.03172787e-01],
       [-1.15518389e+00, -3.91225151e-01, -1.51802703e+00],
       [-1.19494041e+00, -1.25321980e+00, -9.57344040e-01],
       [ 1.60348031e+00,  2.12862042e+00, -1.17055210e+00],
       [ 1.30875249e+00,  9.81586335e-01, -7.94154335e-01],
       [-5.10318286e-01, -1.98245817e+00, -3.62671894e-01],
       [-8.70410759e-01,  1.76648926e-01, -1.97356816e+00],
       [-9.47630358e-01,  4.63733194e-01, -1.44027560e+00],
       [ 1.06821313e+00, -1.03687405e+00,  9.51366145e-01],
       [-9.31853964e-01,  1.67082426e-01, -1.85124672e+00],
       [-1.55341884e+00, -2.77288650e+00, -1.13373829e+00],
       [ 1.34767635e+00, -1.57379170e+00,  2.46898168e+00],
       [ 1.06175103e-01, -1.33955954e+00, -4.90679704e-01],
       [ 1.74138524e+00, -1.80486202e+00,  3.18717060e+00],
       [ 1.77083787e+00, -2.77099968e+00,  4.88470850e-01],
       [-1.32462830e+00,  2.08095368e+00,  3.37087234e+00],
       [ 5.17519086e-01,  1.31025293e+00, -1.05392751e+00],
       [-1.46089438e+00, -2.76172255e+00, -1.04578106e+00],
       [-1.01634535e+00,  1.12853696e+00,  2.40731399e-01],
       [ 3.22897738e-01,  2.64657736e+00, -1.24987067e+00],
       [-1.31004380e-01,  5.95892311e-01, -2.55494907e+00],
       [-5.02824027e-01,  8.39897697e-01, -1.60476098e+00],
       [-2.09117407e-01, -2.39487820e+00, -1.91672293e-01],
       [-1.66833143e+00,  1.94888112e+00,  4.44820644e+00],
       [ 7.94271595e-01,  3.19738317e-01, -1.72214616e+00],
       [-5.03158643e-01, -2.35126002e+00, -3.33633111e-01],
       [ 4.37068743e-01,  1.67898988e+00, -9.35049078e-01],
       [ 2.13890468e-01,  5.80761938e-01, -2.47428905e+00],
       [ 6.72312909e-01, -1.22918182e+00,  1.82774593e-01],
       [ 6.65480227e-01,  9.72093434e-01, -1.26381264e+00],
       [-1.01722413e+00,  1.30756847e+00,  6.73268767e-01],
       [ 6.62740642e-01, -1.09084912e+00, -6.97126604e-02],
       [ 8.24769593e-01, -8.48758013e-01, -1.58920829e-01],
       [-1.89382610e+00,  2.25962310e-02, -1.31048091e+00],
       [ 1.89714687e+00, -1.01904436e+00,  2.77117800e+00],
       [ 1.66274006e+00, -1.27306970e+00,  3.14153855e+00],
       [-7.19478554e-01, -2.84188304e+00, -4.86968272e-01],
       [-1.35184817e-01, -7.37660313e-01, -1.93354051e+00],
       [-1.16252253e+00,  1.56227062e+00,  1.84241338e+00],
       [ 4.33011465e-02, -2.43315801e+00, -8.35349944e-02],
       [ 1.15599085e+00, -2.37429434e+00,  8.62797716e-01],
       [-2.60271765e-01,  2.92035190e-04, -3.21837228e+00],
       [ 7.15535424e-01,  8.55586701e-01, -1.33840927e+00],
       [-1.85562547e+00, -1.52842923e+00, -1.42450765e+00],
       [ 1.80711655e+00,  7.19001666e-01, -7.17749077e-01],
       [ 5.82955997e-01,  4.69995659e-01, -2.06097268e+00],
       [ 7.98567932e-01,  1.51694824e+00, -8.85010339e-01],
       [ 1.26832907e-02,  1.35985796e+00, -1.12025611e+00],
       [ 6.50326687e-01,  1.67389171e+00, -9.06595744e-01],
       [ 4.71161370e-01,  9.45278877e-01, -1.49674455e+00],
       [ 1.02914448e+00,  2.64266640e+00, -1.29430782e+00],
       [ 1.03073979e+00,  1.63288541e+00, -8.68497114e-01],
       [ 7.07596908e-01,  1.65288377e+00, -8.98112660e-01],
       [ 1.90964511e+00, -2.17791106e+00,  2.21019391e+00],
       [-3.14593817e-01, -2.63340434e+00, -2.52980155e-01],
       [-1.84695569e+00,  9.23694082e-02, -1.19408596e+00],
       [ 2.82965311e-01,  2.22541951e+00, -1.03622200e+00],
       [-1.54713712e+00, -1.71869325e+00, -1.10687669e+00],
       [-1.11951575e+00, -1.08918841e+00, -1.02400548e+00],
       [-4.11187724e-01, -1.69003895e-01, -2.90766480e+00],
       [-1.96539905e+00, -5.19945257e-01, -1.63188021e+00],
       [ 4.63589123e-01, -1.14736320e+00, -3.70894257e-01],
       [-1.26783615e+00,  4.22220685e-01, -9.00332499e-01],
       [-8.58875043e-01,  2.35855806e+00,  9.97388448e-01],
       [ 1.90780528e+00,  2.93029626e+00, -1.73210053e+00],
       [ 9.38107640e-01, -9.14436152e-01,  3.04073782e-01],
       [-1.32586087e+00,  8.87629983e-01,  4.37329613e-01],
       [ 1.59673003e+00, -8.61782347e-01,  1.91925904e+00],
       [ 1.72092874e+00, -2.10138607e+00,  2.36819687e+00],
       [ 1.67246670e+00, -2.65713298e+00,  7.26803348e-01],
       [-3.29922228e-02, -1.53635466e+00, -3.52320594e-01],
       [-1.41583611e+00, -5.87054280e-01, -1.33122625e+00],
       [-1.93139599e+00, -1.92532330e+00, -1.48929002e+00],
       [-1.05460131e+00,  9.89367074e-01,  1.81055740e-03],
       [-4.57226115e-01,  2.22807957e+00, -3.23860519e-01],
       [ 2.21375213e-01,  2.21908403e+00, -1.02324094e+00],
       [ 1.78635746e+00,  5.97875498e-01, -6.69408051e-01],
       [-5.79894247e-02, -2.06467548e+00, -1.30942758e-01],
       [-1.91112773e+00, -5.43639722e-01, -1.58342664e+00],
       [ 1.42454194e+00,  1.68085514e+00, -9.24162004e-01],
       [-1.62022960e+00,  2.80007265e-01, -8.47539794e-01],
       [-1.73059490e-01, -8.72856831e-01, -1.57800267e+00],
       [ 6.00221876e-01,  1.24484146e+00, -1.05953762e+00],
       [-8.17191762e-01,  1.62049235e+00,  5.16161729e-01],
       [ 2.30241767e-01, -2.90144650e+00, -8.06505420e-02],
       [-2.84723242e-01, -1.60106479e+00, -4.16099383e-01],
       [ 7.51999350e-01, -2.52082500e+00,  2.75946921e-01],
       [-4.14330760e-02,  1.21105794e+00, -1.29572993e+00],
       [-1.88768778e+00,  2.88502813e+00,  1.97857466e+00],
       [ 1.71231857e+00, -2.50938345e+00,  1.10894501e+00],
       [ 1.45263436e+00,  2.40274786e+00, -1.25788298e+00],
       [-1.16821714e+00,  1.68727286e+00,  2.12804263e+00],
       [ 1.84565311e+00, -1.21381681e-01,  1.93974490e-02],
       [ 4.01057929e-01, -1.98541675e+00,  1.72814447e-01],
       [ 4.09251107e-01,  1.26828015e+00, -1.13765195e+00],
       [-7.27993250e-01, -6.76031986e-01, -1.65672957e+00],
       [-9.93332218e-01,  1.71375967e+00,  1.36403026e+00],
       [ 1.82050294e-01, -8.07860616e-01, -1.56408144e+00],
       [ 1.44388280e+00, -2.62988753e+00,  6.85524791e-01],
       [ 1.41009576e+00,  2.11950474e+00, -1.10496718e+00],
       [ 1.38574283e+00,  2.07251297e+00, -1.07657327e+00],
       [-1.47246388e+00,  1.80294021e-01, -1.07656333e+00],
       [ 4.54195809e-02,  1.19544394e+00, -1.32732440e+00],
       [-1.42594800e+00,  8.81589193e-01,  6.41813635e-01],
       [-1.21636722e+00, -2.99047841e+00, -8.58110411e-01],
       [-5.77767051e-01, -2.69379626e+00, -3.90851143e-01],
       [ 1.85041914e+00, -6.52495919e-01,  1.47561716e+00],
       [-1.75982014e+00,  2.27741338e+00,  4.43595264e+00],
       [-1.33672023e+00, -9.49699525e-01, -1.15688736e+00],
       [-5.56308508e-01, -2.44549397e-01, -2.57424730e+00],
       [-3.19988858e-02, -2.43936465e+00, -1.16313072e-01],
       [ 1.16043414e+00,  9.93746382e-01, -8.52440974e-01],
       [-1.13826672e+00, -2.57654788e+00, -7.56070703e-01],
       [ 7.44410168e-01, -1.32120984e-01, -1.76912542e+00],
       [-1.98012765e-01,  1.98468123e+00, -7.28489212e-01],
       [-1.76047228e+00,  6.01472541e-01,  2.72195753e-02],
       [-1.44384721e+00, -2.75211422e+00, -1.02926626e+00],
       [ 9.32537607e-01,  1.02255731e+00, -9.84681557e-01],
       [-7.13423721e-01,  1.20268723e+00, -4.62246887e-01],
       [ 9.05667013e-01, -8.81367952e-02, -1.36512097e+00],
       [-1.65747914e+00,  6.22194997e-01,  6.06147778e-02],
       [ 1.21832240e+00,  1.33531957e-01, -8.32908055e-01],
       [-6.53085571e-01,  1.82872082e+00,  1.63624358e-01],
       [ 1.04984508e+00,  2.23302230e+00, -1.08635868e+00],
       [-1.17692505e+00,  6.48043868e-01, -5.71150282e-01],
       [ 1.12574833e+00, -2.17323966e+00,  1.14091981e+00],
       [ 1.65236750e+00, -7.79048383e-01,  1.73749614e+00],
       [-1.56520201e+00,  2.61776135e+00,  3.30838832e+00],
       [ 1.54151906e+00,  1.49452610e+00, -8.87874375e-01],
       [ 3.64904104e-01,  2.60800797e+00, -1.23221956e+00],
       [ 5.53842546e-01, -1.60558681e+00,  3.06485955e-01],
       [-1.40360625e-02, -9.91855809e-01, -1.24147933e+00],
       [-1.39732616e+00, -1.92145077e+00, -9.61055247e-01],
       [-1.25513259e+00, -1.00356166e+00, -1.10590235e+00],
       [-1.82942846e+00,  2.72895097e+00,  2.88544525e+00],
       [ 4.29155438e-01, -5.52812169e-01, -1.83935075e+00],
       [ 1.23418496e+00, -3.07361637e-01, -2.77746176e-01],
       [-4.06751593e-01, -1.50762673e-01, -2.93196849e+00],
       [-5.30525063e-01,  2.53925470e+00, -3.26289945e-01],
       [ 1.32471333e+00,  2.19967260e+00, -1.12119359e+00],
       [-1.60345952e+00,  9.37445357e-03, -1.22753634e+00],
       [ 1.94843638e+00, -2.82020137e+00,  3.43010067e-01],
       [-1.87624493e+00,  1.68312491e+00,  4.11362777e+00],
       [-1.02836827e+00,  8.15042376e-01, -5.08775433e-01],
       [-1.93308468e+00,  2.52138337e-01, -9.82398520e-01],
       [-6.25535291e-01, -5.20026010e-01, -2.04798133e+00],
       [-3.40277048e-01,  9.30903272e-01, -1.62938447e+00],
       [ 1.62456601e+00,  5.69654206e-01, -6.61993011e-01],
       [-1.02579309e+00,  9.72776024e-01, -1.22027698e-01],
       [-9.69877460e-01, -2.69538868e+00, -6.39061458e-01],
       [-1.26414715e+00, -2.99675774e+00, -8.98411006e-01],
       [ 1.63511979e+00, -1.33697048e+00,  3.17880959e+00],
       [ 2.31087994e-01, -2.15727589e+00,  3.21696039e-02],
       [-1.10657745e+00, -8.60804139e-01, -1.19536215e+00],
       [ 1.09207101e+00, -1.11792906e+00,  1.18390622e+00],
       [ 3.34575450e-01, -1.22321148e+00, -4.35718640e-01],
       [ 1.06613656e+00,  1.81454578e+00, -9.17534505e-01],
       [ 3.60099090e-01, -1.32856830e+00, -2.40624175e-01],
       [ 1.99710005e+00,  2.27903575e+00, -1.42072469e+00],
       [ 1.48624007e-01,  2.38640674e+00, -1.07716109e+00],
       [-9.68999071e-01,  2.35468251e+00,  1.50983809e+00],
       [-1.17729746e+00, -1.82910546e+00, -7.91381799e-01],
       [-4.98832722e-01, -1.14012699e+00, -9.88764840e-01],
       [-1.13478302e+00,  2.63514652e-01, -1.36910106e+00],
       [ 1.39592683e+00,  2.31252049e+00, -1.19532342e+00]]))


                    
#Build a two block decomposition method class to handle everything
class Two_block_decomposition:
    def __init__(self, N=None, rho=None, sig=None, K=None, seed=None):
      '''
      ######################################
                Hyper parameters 
      ######################################
      '''
      # N: The number of the neurons in the hidden layer 
      self.sig, self.rho, self.N = sig, rho, N 

      # The number of the iterations 
      self.K = K

      # The number of the functions evaluated 
      self.nfunc = 0 

      # The number of the gradients evaluated 
      self.ngrad = 0

      # The number of the outer iterations 
      # As due to the stopping early criteria the model may stop sooner 
      self.iterations = 0 

      # The seed that we want to use for the random number generation
      self.seed = seed

      # The error on the training set before starting the procedure 
      self.baseError = 0 

    def load_parameters(self, W, B, V): 

      # Preload the paramters 
      self.w, self.b, self.v = W, B, V

      # Set the number of the features (n)
      self.n = self.w.shape[0]

    #hyperbolic tangent function, the activation function
    def g(self, t, sig):
      return (np.exp(2 * sig * t) - 1)/(np.exp(2 * sig * t) + 1)
    
    # Loss when we have the hidden layer weights, bias fixed
    def loss_w2(self, x0): 
      self.v = x0
      Loss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2) + self.rho/2 * np.sum(self.v**2)
      loss = Loss(self.x_train, self.y_train)
      return loss

    # Loss when we have the output layer weights fixed
    def loss_w1_bias(self, x0): 
      
      # N: the number of the neurons 
      # n: the number of the features 
      N, n = self.N, self.n

      # Get the parameters of the model 
      self.w, self.b = x0[:n * N].reshape((n, N)), x0[n * N:N + N * n]
      Loss = lambda x, y: np.sum(1/(2*x.shape[0])*(self.pred(x)-y)**2) + self.rho/2 * (np.sum(self.w**2)+np.sum(self.b**2))
      loss = Loss(self.x_train, self.y_train)
      return loss

    # Computes the gradient of the weights in the hidden layer and the bias with respect to the loss function
    def dRegLoss_w1_bias(self, x0): 
      
      # N: the number of the neurons in the hidden layer 
      # n: the number of the features in the training data 
      n, N = self.n, self.N

      # Take the current values of the parameters of the network 
      # The weights of the hidden layer, the bias added, the weights of the ouput layer 
      w, b, v = x0[:n * N].reshape((n, N)), x0[n * N:N + N * n], self.v

      # Take the number of the samples that we have in the training data
      P = self.x_train.shape[0]
      x_train = self.x_train.reshape((P, 1, n))
      y_train = self.y_train.reshape((P, 1, 1))

      # Get the output of the first hidden layer, the multiplication of the weights + the bias of the neuron 
      t = x_train @ w + b

      # Get the prediction of the model, the output of the activation function and the output weights v
      y_pred = self.g(t, self.sig) @ v
      y_pred = y_pred.reshape((P,1,1))

      # Computing the gradient of the parameters with repspect to the loss function
      grad_J = 1
      grad_y_pred = 1 / P * (y_pred - y_train) * grad_J
      grad_g = grad_y_pred @ v.reshape((1, N))
      grad_t = grad_g * (4 * self.sig * np.exp(2 * self.sig * t) / (np.exp(2 * self.sig * t) + 1) ** 2)
      grad_b, grad_mul = grad_t, grad_t
      grad_w = np.transpose(x_train, axes=(0,2,1)) @ grad_mul
      grad_w, grad_b= np.sum(grad_w, axis=0)+self.rho * w, np.sum(grad_b, axis=0)[0]+self.rho * b
      return np.concatenate((grad_w.flatten(), grad_b.flatten()))

    # Two block decomposition method 
    def TBD(self): 
      
       # The number of the functions evaluated 
      self.nfunc = 0 

      # The number of the gradients evaluated 
      self.ngrad = 0

      # N: the number of the neurons in the hidden layer 
      # n: the number of hte features in the training data 
      N, n = self.N, self.n

      # Keep the previous values of the parameters to have them in case the error on the validation set increases 
      prev_w, prev_b, prev_v, prev_val_acc = self.w, self.b, self.v, self.error(self.pred(self.x_val), self.y_val)

      # Iterate K times to update the parameters of the network
      for iter in range(self.K): 
        
        # 1 - Keeping the weights fixed and update the output weights 
        # Updating the values of the output weights 
        Bounds = [(-100, 100)] * self.v.shape[0]
        res = scipy.optimize.dual_annealing(func = self.loss_w2, bounds = Bounds, maxiter=500)
        self.v = res.x

        # How many functions have been evaluated 
        self.nfunc += res.nfev
        # How many gradients have been computed 
        self.ngrad += res.njev
        
        # 2- Keeping the output weights fixed and update the hidden layer weights and bias
        # Updating the values of hidden layer weights and the bias 
        x0 = np.concatenate((self.w.flatten(), self.b.flatten()))
        res = minimize(self.loss_w1_bias, x0, method='BFGS', jac=self.dRegLoss_w1_bias, options= {"maxiter":2000, "disp":False}, 
                      tol=1e-8)
        self.w, self.b = res.x[:n*N].reshape((n, N)), res.x[n*N:N+N*n]

        # How many functions have been evaluated 
        self.nfunc += res.nfev
        # How many gradients have been computed 
        self.ngrad += res.njev

        # Get the error of the model on the validation set 
        current_val_acc = self.error(self.pred(self.x_val), self.y_val)

        # Number of the performed iterations
        self.iterations+=1
        
        # If the error on the validation set got increased, terminate the process and use the previous parameters  
        if current_val_acc >= prev_val_acc: 
          # Put the previous weights as the weights of the model 
          self.w, self.b, self.v = prev_w, prev_b, prev_v
          return True
        
        else: 
          # Keep track of the previous values of the parameters
          prev_w, prev_b, prev_v= self.w, self.b, self.v
          
          # Keep track of the previous error of the model on the validation set 
          prev_val_acc = current_val_acc        
      
      # Return trained successfully
      return True 

    # Building the model using the training data 
    def fit(self, X_train, y_train): 

      '''
      ######################################
                Receiving the data
      ######################################
      '''
      # The training and the test set 
      self.x_train, self.y_train = X_train, y_train

      # Take 20% of the training data as validation set for the stopping criteria 
      
      self.Take_val(0.2)

      self.x_train, self.y_train , self.x_val, self.y_val = self.x_train.to_numpy(), self.y_train.to_numpy(), self.x_val.to_numpy(), self.y_val.to_numpy()
      '''
      ######################################
              Setting the parameters 
      ######################################
      '''

      # The number of the features in the trainig set 
      self.n = self.x_train.shape[1]

      # The randomized parameters of the network 
      # The weights of the hidden layer, the bias, the weights of the output layer 
      self.w, self.b, self.v = np.random.randn(self.n, self.N), np.repeat(0.01, self.N), np.random.randn(self.N) 

      # Take the error of the model before starting the optimization process 
      self.baseError = self.error(self.pred(X_train), y_train)

      # Updating the parameters of the model using the provided data 
      return self.TBD()

    # The forward pass to compute the predictions of the samples
    def pred(self, x): 
      t = x.dot(self.w) + self.b
      return self.g(t, self.sig).dot(self.v)

    def error(self, y_pred, y_true): 
      loss = lambda x, y: np.sum(1/(2*x.shape[0])*(x-y)**2)
      return loss(y_pred, y_true)

    def plotApproximationFunc(self, title = 'Two block decomposition', save_fig = True, path = 'TBD'):
      fig = plt.figure()
      ax = plt.axes(projection='3d')
      x, y = np.linspace(-2, 2, 50), np.linspace(-3, 3, 50)
      X, Y = np.meshgrid(x, y)
      Z = self.approximationFunction(X, Y)
      ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
      ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
      ax.set_title(f'{title} plot')
      if save_fig: 
        plt.savefig(f'{path}.png')
      plt.show()

    def approximationFunction(self, x, y):
      x, y = x.flatten(), y.flatten()
      Z = np.zeros(50*50)
      for i in range(50*50):
          Z[i] = self.pred(np.array([x[i], y[i]]))
      Z = Z.reshape((50,50))
      return Z

    # Storing the value of the parameters of the network
    def save_optimized_params(self, path = 'TBD_params'):
      with open(f'{path}.pkl', 'wb') as f:
        pickle.dump([self.w, self.b, self.v], f)

    def save_hyperparams(self, path = 'TBD_hyperparam'): 
      with open(f'{path}.pkl', 'wb') as f: 
        pickle.dump([self.N, self.rho, self.sig], f)
    
    # Giving some statistics about the last training 
    def last_train_stat_full(self): 
      print(f'The number of the neurons in the hidden layer: \'{self.N}\'')
      print(f'The value of sigma: \'{self.sig}\'')
      print(f'The value of rho: \'{self.rho}\'')
      print(f'The chosen optimization solver: \'BFGS\'')
      print(f'The number of the outer iterations (subproblems solved): \'{self.iterations}\'')
      print(f'The number of the evaluated functions: \'{self.nfunc}\'')
      print(f'The number of the evaluated gradients: \'{self.ngrad}\'')
      print(f'The error of the model before starting the optimization process: \'{self.baseError}\'')
      return [self.N, self.sig, self.rho, 'BFGS', self.iterations, self.nfunc, self.ngrad, self.baseError]

    def last_train_stat_optimize(self): 
      print(f'The number of the outer iterations (subproblems solved): \'{self.iterations}\'')
      print(f'The number of the evaluated functions: \'{self.nfunc}\'')
      print(f'The number of the evaluated gradients: \'{self.ngrad}\'')
      print(f'The error of the model before starting the optimization process: \'{self.baseError}\'')
      return [self.iterations, self.nfunc, self.ngrad, self.baseError]
  
    # Take a portion of the data as the validation set for the stopping criteri
    def Take_val(self, portion): 
      X, y = self.x_train, self.y_train
      np.random.seed(self.seed)
      idx = np.random.choice(X.shape[0],int((1 - portion) *X.shape[0]),replace=False)
      self.x_train, self.y_train, self.x_val, self.y_val = X[X.index.isin(idx)].reset_index(drop=True), y[y.index.isin(idx)].reset_index(drop=True),\
                                   X[~X.index.isin(idx)].reset_index(drop=True), y[~y.index.isin(idx)].reset_index(drop=True)
if __name__ == '__main__': 


	# The matricola of one of the groupmmates
	Seed = 2027647	
	X, y = df.iloc[:,[0,1]], df.iloc[:,2]

	# The hyper parameters of the best model that we got in the Question 1			                   
	N, rho, sig = [35, 1e-05, 1.4]

	# The number of the iterations in the two block decomposition approach
	K = 20

	# Create a Two block decomposition approach object
	TBD_obj = Two_block_decomposition(N=N, rho=rho, sig=sig, K=K, seed=Seed)

	# Begin the optimizing the network's parameters 
	opt = TBD_obj.fit(X, y)

	# Check the error on the training set 
	trErr = TBD_obj.error(TBD_obj.pred(X), y)
	print(f"Training Error: \'{round(trErr, 5)}\'")
	TBD_obj.save_optimized_params('Bonus_params')
	TBD_obj.save_hyperparams('Bonus_hyperparam')
