//
//  ConvLayer.cpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "ConvLayer.hpp"
#include "PoolLayer.hpp"
#include "FcLayer.hpp"
#include <cmath>
#include <numeric>
#include <random>

ConvLayer::ConvLayer(const int kernelNum, const int kernelSize, const int convStride, const convSet& convType, const activationSet& activationFunction, Layer* prev) : kernelNum(kernelNum), kernelSize(kernelSize), convStride(convStride), convType(convType), activationFunction(activationFunction), Layer(prev){
    assert(kernelNum > 0);
    assert(kernelSize > 0);
    assert(convStride > 0);
    assert(kernelSize % 2 == 1);
    layerType = layerSet::CONV;
}

void ConvLayer::initialize(){
    default_random_engine e;
    normal_distribution<float> n(1.f / kernelSize, 0.5 / (kernelNum * kernelSize));
    kernel.resize(kernelNum);
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < kernelSize; j++){
            kernel[i].push_back(n(e));
        }
        bias.push_back(n(e));
    }
}

void ConvLayer::convFull(){
    vector<float> paddingInput;
    
    for (int i = 0; i < kernelSize - 1; i++){
        paddingInput.push_back(0);
    }
    copy(prev->out.begin(), prev->out.end(), back_inserter(paddingInput));
    for (int i = 0; i < kernelSize - 1; i++){
        paddingInput.push_back(0);
    }
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < int(prev->out.size()) + kernelSize - 1; j += convStride){
            out.push_back(inner_product(kernel[i].rbegin(), kernel[i].rend(), paddingInput.begin() + j, bias[i]));
        }
    }
}


void ConvLayer::convSame(){
    vector<float> paddingInput;
    
    for (int i = 0; i < (kernelSize - 1) / 2; i++){
        paddingInput.push_back(0);
    }
    copy(prev->out.begin(), prev->out.end(), back_inserter(paddingInput));
    for (int i = 0; i < (kernelSize - 1) / 2; i++){
        paddingInput.push_back(0);
    }
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < int(prev->out.size()); j += convStride){
            out.push_back(inner_product(kernel[i].rbegin(), kernel[i].rend(), paddingInput.begin() + j, bias[i]));
        }
    }
}

void ConvLayer::convValid(){
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < int(prev->out.size()) - kernelSize + 1; j += convStride){
            out.push_back(inner_product(kernel[i].rbegin(), kernel[i].rend(), prev->out.begin() + j, bias[i]));
        }
    }    
}

void ConvLayer::activationSigmoid(){
    for_each(out.begin(), out.end(), [](float &o) {o = 1.0 / (exp(-o) + 1);});
}

void ConvLayer::activationTanh(){
    for_each(out.begin(), out.end(), [](float &o) {o = std::tanh(o);});
}

void ConvLayer::activationReLU(){
    for_each(out.begin(), out.end(), [](float &o) {o = (o > 0)? o: 0;});
}

void ConvLayer::activationLeakyReLU(){
    for_each(out.begin(), out.end(), [](float &o) {o = (o > 0)? o: 0.01 * o;});
}

void ConvLayer::nodeNumComputation(){
    int actualNodeNum = prev->nodeNum;
    switch (convType) {
        case convSet::full:
            actualNodeNum += 2 * (kernelSize - 1);
            break;
        case convSet::same:
            actualNodeNum += kernelSize - 1;
            break;
        case convSet::valid:
            break;
        default:
            cout << "[Error@ConvLayer:check]: Unable to handle conv type " << convType << endl;
            exit(1);
            break;
    }
    int numerator = actualNodeNum - kernelSize;
    if ((numerator >= 0) && (numerator % convStride == 0)){
        nodeNum = kernelNum * (numerator / convStride + 1);
    }
    else {
        if (prev->layerType == Layer::INPUT){
            int padding = 0;
            if (numerator < 0){
                padding = -numerator;
                numerator = 0;
            }
            else if (numerator % convStride != 0){
                padding = kernelSize + ceil((float)numerator / convStride) * convStride - actualNodeNum;
            }
            cout << "[Warning@ConvLayer:check]: Incompatible conv border\nThe node num of input layer changed(From " << prev->nodeNum << " to " << prev->nodeNum + padding << ")." << endl;
            prev->nodeNum += padding;
            for (int i = 0; i < padding; i++)
                prev->out.push_back(0);
            nodeNum = kernelNum * int(ceil(numerator / convStride) + 1);
        }
        else {
            cout << "[Error@ConvLayer:check]: Incompatible conv border" << endl;
            exit(1);
        }
    }
}

void ConvLayer::forward(){
    out.clear();
    switch (convType) {
        case convSet::full:
            convFull();
            break;
        case convSet::same:
            convSame();
            break;
        case convSet::valid:
            convValid();
            break;
        default:
            cout << "[Error@ConvLayer:forward]: Unable to handle conv type " << convType << endl;
            exit(1);
            break;
    }
    switch (activationFunction) {
        case activationSet::sigmoid:
            activationSigmoid();
            break;
        case activationSet::tanh:
            activationTanh();
            break;
        case activationSet::ReLU:
            activationReLU();
            break;
        case activationSet::leakyReLU:
            activationLeakyReLU();
            break;
        default:
            cout << "[Error@ConvLayer:forward]: Unable to handle activation function " << activationFunction << endl;
            exit(1);
            break;
    }
}

void ConvLayer::coefFromPoolGeneral(){
    PoolLayer* poolPtr = (PoolLayer*)next;
    
    for (int i = 0; i < poolPtr->nodeNum; i++){
        for (int j = 0; j < min(poolPtr->poolSize, nodeNum - i * poolPtr->poolSize); j++){
            coef.push_back(poolPtr->coef[i] * poolPtr->weight[i][j]);
        }
    }
}

void ConvLayer::coefFromPoolOverlap(){
    PoolLayer* poolPtr = (PoolLayer*)next;
    
    coef.resize(nodeNum, 0);
    
    for (int i = 0; i < nodeNum; i++){
        for (int j = 0; j < poolPtr->nodeNum; j++){
            if ((i - j * poolPtr->poolStride >= 0) && (poolPtr->weight[j].size() > i - j * poolPtr->poolStride)){
                coef[i] += poolPtr->coef[j] * poolPtr->weight[j][i - j * poolPtr->poolStride];
            }
        }
    }
}

void ConvLayer::coefFromFc(){
    FcLayer* fcPtr = (FcLayer*)next;
    
    for (int i = 0; i < nodeNum; i++){
        float oneCoef = 0.f;
        for (int j = 0; j < fcPtr->coef.size(); j++){
            oneCoef += fcPtr->coef[j] * fcPtr->weight[j][i];
        }
        coef.push_back(oneCoef);
    }
}

void ConvLayer::coefFromConvFull(){
    ConvLayer* convPtr = (ConvLayer*)next;
    vector<vector<float>> paddingCoef;
    for (int j = 0; j < convPtr->kernel.size(); j++){
        vector<float> onePaddingCoef;
        for (int i = 0; i < convPtr->coef.size() / convPtr->kernel.size(); i++){
            onePaddingCoef.push_back(convPtr->coef[j * convPtr->coef.size() / convPtr->kernel.size() + i]);
            if (i != convPtr->coef.size() / convPtr->kernel.size() - 1){
                for (int l = 0; l < convPtr->convStride - 1; l++){
                    onePaddingCoef.push_back(0);
                }
            }
        }
        
        paddingCoef.push_back(onePaddingCoef);
    }
    for (int i = 0; i < nodeNum; i++){
        float oneCoef = 0.f;
        for (int j = 0; j < convPtr->kernel.size(); j++){
            oneCoef += inner_product(convPtr->kernel[j].begin(), convPtr->kernel[j].end(), paddingCoef[j].begin() + i, 0.f);
        }
        coef.push_back(oneCoef);
    }
}

void ConvLayer::coefFromConvSame(){
    ConvLayer* convPtr = (ConvLayer*)next;
    vector<vector<float>> paddingCoef;
    for (int j = 0; j < convPtr->kernel.size(); j++){
        vector<float> onePaddingCoef;
        for (int i = 0; i < (convPtr->kernel[j].size() - 1) / 2; i++)
            onePaddingCoef.push_back(0);
        for (int i = 0; i < convPtr->coef.size() / convPtr->kernel.size(); i++){
            onePaddingCoef.push_back(convPtr->coef[j * convPtr->coef.size() / convPtr->kernel.size() + i]);
            if (i != convPtr->coef.size() / convPtr->kernel.size() - 1){
                for (int l = 0; l < convPtr->convStride - 1; l++){
                    onePaddingCoef.push_back(0);
                }
            }
        }
        for (int i = 0; i < (convPtr->kernel[j].size() - 1) / 2; i++)
            onePaddingCoef.push_back(0);
        
        paddingCoef.push_back(onePaddingCoef);
    }
    for (int i = 0; i < nodeNum; i++){
        float oneCoef = 0.f;
        for (int j = 0; j < convPtr->kernel.size(); j++){
            oneCoef += inner_product(convPtr->kernel[j].begin(), convPtr->kernel[j].end(), paddingCoef[j].begin() + i, 0.f);
        }
        coef.push_back(oneCoef);
    }
}

void ConvLayer::coefFromConvValid(){
    ConvLayer* convPtr = (ConvLayer*)next;
    vector<vector<float>> paddingCoef;
    for (int j = 0; j < convPtr->kernel.size(); j++){
        vector<float> onePaddingCoef;
        for (int i = 0; i < convPtr->kernel[j].size() - 1; i++)
            onePaddingCoef.push_back(0);
        for (int i = 0; i < convPtr->coef.size() / convPtr->kernel.size(); i++){
            onePaddingCoef.push_back(convPtr->coef[j * convPtr->coef.size() / convPtr->kernel.size() + i]);
            if (i != convPtr->coef.size() / convPtr->kernel.size() - 1){
                for (int l = 0; l < convPtr->convStride - 1; l++){
                    onePaddingCoef.push_back(0);
                }
            }
        }
        for (int i = 0; i < convPtr->kernel[j].size() - 1; i++)
            onePaddingCoef.push_back(0);
        
        paddingCoef.push_back(onePaddingCoef);
    }
    for (int i = 0; i < nodeNum; i++){
        float oneCoef = 0.f;
        for (int j = 0; j < convPtr->kernel.size(); j++){
            oneCoef += inner_product(convPtr->kernel[j].begin(), convPtr->kernel[j].end(), paddingCoef[j].begin() + i, 0.f);
        }
        coef.push_back(oneCoef);
    }
}

void ConvLayer::coefDerivativeSigmoid(){
    for (int i = 0; i < coef.size(); i++){
        coef[i] *= out[i] * (1 - out[i]);
    }
}

void ConvLayer::coefDerivativeTanh(){
    for (int i = 0; i < coef.size(); i++){
        coef[i] *= 1 - pow(std::tanh(out[i]), 2);
    }
}

void ConvLayer::coefDerivativeReLU(){
    for (int i = 0; i < coef.size(); i++){
        if (out[i] < 0)
            coef[i] = 0;
    }
}

void ConvLayer::coefDerivativeLeakyReLU(){
    for (int i = 0; i < coef.size(); i++){
        if (out[i] < 0)
            coef[i] *= 0.01;
    }
}

void ConvLayer::kernelUpdateConvValid(const float alpha){
    int outNumPerKernel = ceil((float)(prev->nodeNum - kernelSize + 1) / convStride);
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < kernelSize; j++){
            for (int k = 0; k < outNumPerKernel; k++){
                kernel[i][j] -= alpha * coef[i * outNumPerKernel + k] * prev->out[kernelSize - 1 - j + k * convStride];
            }
        }
        bias[i] -= alpha * accumulate(coef.begin() + i * outNumPerKernel, coef.begin() + (i + 1) * outNumPerKernel, 0.f);
    }
}

void ConvLayer::kernelUpdateConvSame(const float alpha){
    vector<float> paddingOut;
    
    for (int i = 0; i < (kernelSize - 1) / 2; i++){
        paddingOut.push_back(0);
    }
    copy(prev->out.begin(), prev->out.end(), back_inserter(paddingOut));
    for (int i = 0; i < (kernelSize - 1) / 2; i++){
        paddingOut.push_back(0);
    }
    int outNumPerKernel = ceil((float)((int)paddingOut.size() - kernelSize + 1) / convStride);
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < kernelSize; j++){
            for (int k = 0; k < outNumPerKernel; k++){
                kernel[i][j] -= alpha * coef[i * outNumPerKernel + k] * paddingOut[kernelSize - 1 - j + k * convStride];
            }
        }
        bias[i] -= alpha * accumulate(coef.begin() + i * outNumPerKernel, coef.begin() + (i + 1) * outNumPerKernel, 0.f);
    }
}

void ConvLayer::kernelUpdateConvFull(const float alpha){
    vector<float> paddingOut;
    
    for (int i = 0; i < kernelSize - 1; i++){
        paddingOut.push_back(0);
    }
    copy(prev->out.begin(), prev->out.end(), back_inserter(paddingOut));
    for (int i = 0; i < kernelSize - 1; i++){
        paddingOut.push_back(0);
    }
    int outNumPerKernel = ceil((float)((int)paddingOut.size() - kernelSize + 1) / convStride);
    for (int i = 0; i < kernelNum; i++){
        for (int j = 0; j < kernelSize; j++){
            for (int k = 0; k < outNumPerKernel; k++){
                kernel[i][j] -= alpha * coef[i * outNumPerKernel + k] * paddingOut[kernelSize - 1 - j + k * convStride];
            }
        }
        bias[i] -= alpha * accumulate(coef.begin() + i * outNumPerKernel, coef.begin() + (i + 1) * outNumPerKernel, 0.f);
    }
}

void ConvLayer::backward(const float alpha){
    coef.clear();
    switch (next->layerType) {
        case layerSet::POOL:
            switch (((PoolLayer*)next)->poolType) {
                case PoolLayer::general:
                    coefFromPoolGeneral();
                    break;
                case PoolLayer::overlap:
                    coefFromPoolOverlap();
                    break;
                default:
                    cout << "[Error@ConvLayer:backward]: Unable to handle pool type " << ((PoolLayer*)next)->poolType << endl;
                    exit(1);
                    break;
            }
            break;
        case layerSet::FC:
            coefFromFc();
            break;
        case layerSet::CONV:
            switch (((ConvLayer*)next)->convType) {
                case ConvLayer::full:
                    coefFromConvFull();
                    break;
                case ConvLayer::same:
                    coefFromConvSame();
                    break;
                case ConvLayer::valid:
                    coefFromConvValid();
                    break;
                default:
                    cout << "[Error@ConvLayer:backward]: Unable to handle conv type " << ((ConvLayer*)next)->convType << endl;
                    exit(1);
                    break;
            }
            break;
        default:
            cout << "[Error@ConvLayer:backward]: Unable to handle layer type " << next->layerType << endl;
            exit(1);
            break;
    }
    
    switch (activationFunction) {
        case activationSet::sigmoid:
            coefDerivativeSigmoid();
            break;
        case activationSet::tanh:
            coefDerivativeTanh();
            break;
        case activationSet::ReLU:
            coefDerivativeReLU();
            break;
        case activationSet::leakyReLU:
            coefDerivativeLeakyReLU();
            break;
        default:
            cout << "[Error@ConvLayer:backward]: Unable to handle activation function " << activationFunction << endl;
            exit(1);
            break;
    }
    
    switch (convType) {
        case convSet::valid:
            kernelUpdateConvValid(alpha);
            break;
        case convSet::same:
            kernelUpdateConvSame(alpha);
            break;
        case convSet::full:
            kernelUpdateConvFull(alpha);
            break;
        default:
            cout << "[Error@ConvLayer:backward]: Unable to handle conv type " << convType << endl;
            exit(1);
            break;
    }
    
    
//    for_each(kernel[0].begin(), kernel[0].end(), [](float o) -> void {cout << o << " ";});
//    cout << endl;
}
