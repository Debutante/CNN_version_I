//
//  FcLayer.cpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "FcLayer.hpp"
#include "PoolLayer.hpp"
#include "ConvLayer.hpp"
#include <numeric>
#include <cmath>
#include <assert.h>
#include <random>

FcLayer::FcLayer(const int nodeNum, const activationSet& activationFunction, Layer* prev) : Layer(nodeNum, prev), activationFunction(activationFunction){
    layerType = layerSet::FC;
    isOutputLayer = false;
}

void FcLayer::initialize(){
    int prevNodeNum = prev->nodeNum;
    
    default_random_engine e;
    normal_distribution<float> n(1.f / prevNodeNum, 0.5 / (nodeNum * prevNodeNum));
    weight.resize(nodeNum);
    for (int i = 0; i < nodeNum; i++){
        for (int j = 0; j < prevNodeNum; j++){
            weight[i].push_back(n(e));
        }
        bias.push_back(n(e));
    }
}

void FcLayer::activationSigmoid(){
    for_each(out.begin(), out.end(), [](float &o) {o = 1.0 / (exp(-o) + 1);});
}

void FcLayer::activationTanh(){
    for_each(out.begin(), out.end(), [](float &o) {o = std::tanh(o);});
}

void FcLayer::activationReLU(){
    for_each(out.begin(), out.end(), [](float &o) {o = (o > 0)? o: 0;});
}

void FcLayer::activationLeakyReLU(){
    for_each(out.begin(), out.end(), [](float &o) {o = (o > 0)? o: 0.01 * o;});
}

void FcLayer::activationSoftmax(){
    float denominator = accumulate(out.begin(), out.end(), 0.f, [](float x, float y) {return x + exp(y);});
    for_each(out.begin(), out.end(), [denominator](float &o) {o = exp(o) / denominator;});
}

void FcLayer::forward(){
    out.clear();
    for (int j = 0; j < nodeNum; j++){
        out.push_back(inner_product(prev->out.begin(), prev->out.end(), weight[j].begin(), bias[j]));
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
        case activationSet::softmax:
            activationSoftmax();
            break;
        default:
            cout << "[Error@FcLayer:forward]: Unable to handle activation function " << activationFunction << endl;
            exit(3);
            break;
    }
}

void FcLayer::coefDerivativeSigmoid(){
    for (int i = 0; i < coef.size(); i++){
        coef[i] *= out[i] * (1 - out[i]);
    }
}

void FcLayer::coefDerivativeTanh(){
    for (int i = 0; i < coef.size(); i++){
        coef[i] *= 1 - pow(std::tanh(out[i]), 2);
    }
}

void FcLayer::coefDerivativeReLU(){
    for (int i = 0; i < coef.size(); i++){
        if (out[i] < 0)
            coef[i] = 0;
    }
}

void FcLayer::coefDerivativeLeakyReLU(){
    for (int i = 0; i < coef.size(); i++){
        if (out[i] < 0)
            coef[i] *= 0.01;
    }
}

void FcLayer::coefFromFc(){
    FcLayer* fcPtr = (FcLayer*)next;
    
    for (int i = 0; i < nodeNum; i++){
        float oneCoef = 0.f;
        for (int j = 0; j < fcPtr->coef.size(); j++){
            oneCoef += fcPtr->coef[j] * fcPtr->weight[j][i];
        }
        coef.push_back(oneCoef);
    }
}

void FcLayer::coefDerivativeCrossEntropySoftmax(const vector<float>& o){
    for (int i = 0; i < nodeNum; i++){
        coef.push_back(out[i] - o[i]);
    }
}

void FcLayer::coefDerivativeSquared(const vector<float>& o){
    for (int i = 0; i < nodeNum; i++){
        coef.push_back(out[i] - o[i]);
    }
}

void FcLayer::coefDerivativeLogCosh(const vector<float>& o){
    for (int i = 0; i < nodeNum; i++){
        coef.push_back(std::tanh(out[i] - o[i]) * out[i]);
    }
}

void FcLayer::coefFromPoolGeneral(){
    PoolLayer* poolPtr = (PoolLayer*)next;
    
    for (int i = 0; i < poolPtr->nodeNum; i++){
        for (int j = 0; j < min(poolPtr->poolSize, nodeNum - i * poolPtr->poolSize); j++){
            coef.push_back(poolPtr->coef[i] * poolPtr->weight[i][j]);
        }
    }
}

void FcLayer::coefFromPoolOverlap(){
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

void FcLayer::coefFromConvFull(){
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

void FcLayer::coefFromConvSame(){
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

void FcLayer::coefFromConvValid(){
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

void FcLayer::weightUpdate(const float alpha){
    for (int i = 0; i < weight.size(); i++){
        for (int j = 0; j < weight[i].size(); j++){
            weight[i][j] -= alpha * coef[i] * prev->out[j];
        }
        bias[i] -= alpha * coef[i];
    }
}

void FcLayer::backward(const float alpha, const vector<float>& o, const lossSet& lossFunction){
    coef.clear();
    switch (lossFunction) {
        case lossSet::cross_entropy:
            switch (activationFunction) {
                case activationSet::softmax:
                    coefDerivativeCrossEntropySoftmax(o);
                    break;
                default:
                    cout << "[Error@FcLayer:backward_crossEntropy]: Unable to handle activation function " << activationFunction << endl;
                    exit(3);
                    break;
            }
            break;
        case lossSet::squared:
            coefDerivativeSquared(o);
            break;
        case lossSet::log_cosh:
            coefDerivativeLogCosh(o);
            break;
        default:
            cout << "[Error@FcLayer:backward_lossFunction]: Unable to handle loss function " << lossFunction << endl;
            exit(3);
            break;
    }
//    if (lossFunction == lossSet::cross_entropy){
//        switch (activationFunction) {
//            case activationSet::softmax:
//                coefDerivativeCrossEntropySoftmax(o);
//                break;
//            default:
//                cout << "[Error@FcLayer:backward_crossEntropy]: Unable to handle activation function " << activationFunction << endl;
//                exit(3);
//                break;
//        }
//    }
//    else {
//        switch (lossFunction){
//            case lossSet::squared:
//                coefDerivativeSquared(o);
//                break;
//            case lossSet::log_cosh:
//                coefDerivativeLogCosh(o);
//                break;
//            default:
//                cout << "[Error@FcLayer:backward_lossFunction]: Unable to handle loss function " << lossFunction << endl;
//                exit(3);
//                break;
//        }
//    }
    if (lossFunction != lossSet::cross_entropy){
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
                cout << "[Error@FcLayer:backward]: Unable to handle activation function " << activationFunction << endl;
                exit(3);
                break;
        }
    }
    weightUpdate(alpha);
}

void FcLayer::backward(const float alpha){
    coef.clear();
    switch (next->layerType) {
        case layerSet::FC:
            coefFromFc();
            break;
        case layerSet::POOL:
            switch (((PoolLayer*)next)->poolType) {
                case PoolLayer::general:
                    coefFromPoolGeneral();
                    break;
                case PoolLayer::overlap:
                    coefFromPoolOverlap();
                    break;
                default:
                    cout << "[Error@FcLayer:backward]: Unable to handle pool type " << ((PoolLayer*)next)->poolType << endl;
                    exit(3);
                    break;
            }
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
            cout << "[Error@FcLayer:backward]: Unable to handle layer type " << next->layerType << endl;
            exit(3);
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
            cout << "[Error@FcLayer:backward]: Unable to handle activation function " << activationFunction << endl;
            exit(3);
            break;
    }
    
    weightUpdate(alpha);
}
