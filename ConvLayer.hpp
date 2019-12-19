//
//  ConvLayer.hpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef ConvLayer_hpp
#define ConvLayer_hpp

#include "Layer.hpp"

class ConvLayer: public Layer{
public:
    enum convSet {full, same, valid};
    enum activationSet {sigmoid, tanh, ReLU, leakyReLU};
    int kernelNum;
    int kernelSize;
    vector<vector<float>> kernel;
    int convStride;
    convSet convType;//full same valid
    vector<float> bias;
    activationSet activationFunction;
    vector<float> coef;

    ConvLayer(const int, const int, const int, const convSet&, const activationSet&, Layer*);
    void nodeNumComputation();
    void initialize();
    void forward();
    void backward(const float);
    
private:
    void convFull();
    void convSame();
    void convValid();
    void activationSigmoid();
    void activationTanh();
    void activationReLU();
    void activationLeakyReLU();
    void coefFromPoolGeneral();
    void coefFromPoolOverlap();
    void coefFromFc();
    void coefFromConvFull();
    void coefFromConvSame();
    void coefFromConvValid();
    void coefDerivativeSigmoid();
    void coefDerivativeTanh();
    void coefDerivativeReLU();
    void coefDerivativeLeakyReLU();
    void kernelUpdateConvValid(const float);
    void kernelUpdateConvSame(const float);
    void kernelUpdateConvFull(const float);

    
};

#endif /* ConvLayer_hpp */
