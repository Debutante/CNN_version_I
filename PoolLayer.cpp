//
//  PoolLayer.cpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "PoolLayer.hpp"
#include "FcLayer.hpp"
#include "ConvLayer.hpp"
#include <numeric>
#include <cmath>

PoolLayer::PoolLayer(const int poolSize, const poolSet& poolType, const sampleSet& sampleType, Layer* prev) : poolSize(poolSize), poolType(poolType), sampleType(sampleType), Layer(prev){
    assert(poolSize > 0);
    assert(poolType == poolSet::general);
    layerType = layerSet::POOL;
}

PoolLayer::PoolLayer(const int poolSize, const poolSet& poolType, const sampleSet& sampleType, Layer* prev, const int poolStride) : poolSize(poolSize), poolType(poolType), sampleType(sampleType), Layer(prev), poolStride(poolStride){
    assert(poolSize > 0);
    assert(poolType == poolSet::overlap);
    assert(poolStride > 0);
    layerType = layerSet::POOL;
}

void PoolLayer::nodeNumComputation(){
    switch (poolType) {
        case poolSet::general:
            nodeNum = ceil((float)prev->nodeNum / poolSize);
            break;
        case poolSet::overlap:
            if (poolSize >= poolStride)
                nodeNum = ceil((float)(prev->nodeNum - poolSize) / poolStride) + 1;
            else {
                cout << "[Warning@PoolLayer:nodeNumComputation]: Risk of omitting the output of the last layer" << endl;
                nodeNum = (prev->nodeNum - poolSize) / poolStride + 1;
            }
            break;
        default:
            cout << "[Error@PoolLayer:nodeNumComputation]: Unable to handle pool type " << poolType << endl;
            exit(2);
            break;
    }
}

void PoolLayer::coefFromConvFull(){
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

void PoolLayer::coefFromConvSame(){
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

void PoolLayer::coefFromConvValid(){
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

void PoolLayer::maxPool(){
    switch (poolType) {
        case poolSet::general:
            for (int i = 0; i < nodeNum; i++){
                auto max = max_element(prev->out.begin() + i * poolSize, prev->out.begin() + i * poolSize + min(poolSize, prev->nodeNum - i * poolSize));
                out.push_back(*max);
                vector<float> oneWeight(min(poolSize, prev->nodeNum - i * poolSize), 0);
                oneWeight[distance(prev->out.begin() + i * poolSize, max)] = 1;
                weight.push_back(oneWeight);
            }
            break;
        case poolSet::overlap:
            for (int i = 0; i < nodeNum; i++){
                auto max = max_element(prev->out.begin() + i * poolStride, prev->out.begin() + i * poolStride + min(poolSize, prev->nodeNum - i * poolStride));
                out.push_back(*max);
                vector<float> oneWeight(min(poolSize, prev->nodeNum - i * poolStride), 0);
                oneWeight[distance(prev->out.begin() + i * poolStride, max)] = 1;
                weight.push_back(oneWeight);
            }
            break;
        default:
            cout << "[Error@PoolLayer:maxPool]: Unable to handle pool type " << poolType << endl;
            exit(2);
            break;
    }
}

void PoolLayer::avgPool(){
    switch (poolType) {
        case poolSet::general:
            for (int i = 0; i < nodeNum; i++){
                out.push_back(accumulate(prev->out.begin() + i * poolSize, prev->out.begin() + i * poolSize + min(poolSize, prev->nodeNum - i * poolSize), 0.f) / min(poolSize, prev->nodeNum - i * poolSize));
                vector<float> oneWeight(min(poolSize, prev->nodeNum - i * poolSize), 1.f / min(poolSize, prev->nodeNum - i * poolSize));
                weight.push_back(oneWeight);
            }
            break;
        case poolSet::overlap:
            for (int i = 0; i < nodeNum; i++){
                out.push_back(accumulate(prev->out.begin() + i, prev->out.begin() + i + poolSize, 0.f) / min(poolSize, prev->nodeNum - i));
                vector<float> oneWeight(min(poolSize, prev->nodeNum - i * poolStride), 1.f / min(poolSize, prev->nodeNum - i * poolStride));
                weight.push_back(oneWeight);
            }
            break;
        default:
            cout << "[Error@PoolLayer:avgPool]: Unable to handle pool type " << poolType << endl;
            exit(2);
            break;
    }
    
}

void PoolLayer::forward(){
    out.clear();
    weight.clear();
    switch (sampleType) {
        case sampleSet::max:
            maxPool();
            break;
        case sampleSet::average:
            avgPool();
            break;
            
        default:
            cout << "[Error@PoolLayer:forward]: Unable to handle sample type " << sampleType << endl;
            exit(2);
            break;
    }
}

void PoolLayer::coefFromFc(){
    FcLayer* fcPtr = (FcLayer*)next;
    
    for (int i = 0; i < nodeNum; i++){
        float oneCoef = 0.f;
        for (int j = 0; j < fcPtr->coef.size(); j++){
            oneCoef += fcPtr->coef[j] * fcPtr->weight[j][i];
        }
        coef.push_back(oneCoef);
    }
}

void PoolLayer::coefFromPoolOverlap(){
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

void PoolLayer::coefFromPoolGeneral(){
    PoolLayer* poolPtr = (PoolLayer*)next;
    
    for (int i = 0; i < poolPtr->nodeNum; i++){
        for (int j = 0; j < min(poolPtr->poolSize, nodeNum - i * poolPtr->poolSize); j++){
            coef.push_back(poolPtr->coef[i] * poolPtr->weight[i][j]);
        }
    }
}

void PoolLayer::backward(){
    coef.clear();
    switch (next->layerType) {
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
                    cout << "[Error@PoolLayer:backward]: Unable to handle conv type " << ((ConvLayer*)next)->convType << endl;
                    exit(2);
                    break;
            }
            break;
        case layerSet::POOL:
            switch (((PoolLayer*)next)->poolType) {
                case poolSet::overlap:
                    coefFromPoolOverlap();
                    break;
                case poolSet::general:
                    coefFromPoolGeneral();
                    break;
                default:
                    cout << "[Error@PoolLayer:backward]: Unable to handle pool type " << ((PoolLayer*)next)->poolType << endl;
                    exit(2);
                    break;
        }
            break;
        default:
            cout << "[Error@PoolLayer:backward]: Unable to handle layer type " << next->layerType << endl;
            exit(2);
            break;
    }
}
