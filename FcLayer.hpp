//
//  FcLayer.hpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef FcLayer_hpp
#define FcLayer_hpp

#include "Layer.hpp"

class FcLayer: public Layer{
public:
    enum activationSet {sigmoid, tanh, ReLU, leakyReLU, softmax};
    enum lossSet {cross_entropy, squared, log_cosh};
    vector<vector<float>> weight;
    vector<float> bias;
    bool isOutputLayer;
    activationSet activationFunction;
    vector<float> coef;
    
    FcLayer(const int, const activationSet&, Layer*);
    void initialize();
    void forward();
    void backward(const float, const vector<float>&, const lossSet&);
    void backward(const float);
    
private:
    void activationSigmoid();
    void activationTanh();
    void activationReLU();
    void activationLeakyReLU();
    void activationSoftmax();
    void coefDerivativeSigmoid();
    void coefDerivativeTanh();
    void coefDerivativeReLU();
    void coefDerivativeLeakyReLU();
    void coefFromFc();
    void coefDerivativeCrossEntropySoftmax(const vector<float>&);
    void coefDerivativeSquared(const vector<float>&);
    void coefDerivativeLogCosh(const vector<float>&);
    void coefFromPoolGeneral();
    void coefFromPoolOverlap();
    void coefFromConvFull();
    void coefFromConvSame();
    void coefFromConvValid();
    void weightUpdate(const float);
    
};

#endif /* FcLayer_hpp */
