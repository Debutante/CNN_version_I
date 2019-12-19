//
//  Model.cpp
//  test1203
//
//  Created by 许清嘉 on 12/8/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Model.hpp"
#include <assert.h>
#include <random>
#include <numeric>
#include <cmath>

Model::Model(){
    mode = NORMAL;
}

Model::Model(const vector<Layer*>& net) : net(net){
    
}

void Model::addInputLayer(const int nodeNum){
    net.push_back(new Layer(nodeNum, nullptr));
}

void Model::addConvLayer(const int kernelNum, const int kernelSize, const int convStride, const ConvLayer::convSet& convType, const ConvLayer::activationSet& activationFunction){
    net.push_back(new ConvLayer(kernelNum, kernelSize, convStride, convType, activationFunction, net.back()));
    (*(net.end() - 2))->next = net.back();
}

void Model::addPoolLayer(const int poolSize, const PoolLayer::poolSet& poolType, const PoolLayer::sampleSet& sampleType){
        net.push_back(new PoolLayer(poolSize, poolType, sampleType, net.back()));
    (*(net.end() - 2))->next = net.back();
}

void Model::addPoolLayer(const int poolSize, const PoolLayer::poolSet& poolType, const PoolLayer::sampleSet& sampleType, const int poolStride){
        net.push_back(new PoolLayer(poolSize, poolType, sampleType, net.back(), poolStride));
    (*(net.end() - 2))->next = net.back();
}

void Model::addFcLayer(const int nodeNum, const FcLayer::activationSet& activationFunction){
    net.push_back(new FcLayer(nodeNum, activationFunction, net.back()));
    (*(net.end() - 2))->next = net.back();
}


void Model::testInitialize1(){
    cout << "[prompt]: Using test initializer 1" << endl;
    //input
    
    //conv
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0}), vector<float>({0.3, 0, 0.4})};
    ((ConvLayer*)net[1])->bias = {0, 0};

    //fc
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4}), vector<float>({-0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4})};
    ((FcLayer*)net[3])->bias = {0.5, -0.5};
    
    mode = TEST;
}

void Model::testInitialize2(){
    cout << "[prompt]: Using test initializer 2" << endl;
    //input
    
    //conv1
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[1])->bias = {0.5, -0.5};
    
    //conv2
    ((ConvLayer*)net[3])->kernel = {vector<float>({-1, 0, 1}), vector<float>({1, 0, -1}), vector<float>({1, 1, 0})};
    ((ConvLayer*)net[3])->bias = {0, 0, 0};

    //fc
    ((FcLayer*)net[5])->weight = {vector<float>({0.1, 0.2, 0.3}), vector<float>({-0.1, -0.2, -0.3})};
    ((FcLayer*)net[5])->bias = {0.4, -0.4};
    
    mode = TEST;
}

void Model::testInitialize3(){
    cout << "[prompt]: Using test initializer 3" << endl;
    //input
    
    //conv
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[1])->bias = {0.5, -0.5};
    
    //fc1
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[3])->bias = {0.3, -0.3};
    
    //fc2
    ((FcLayer*)net[4])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[4])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize4(){
    cout << "[prompt]: Using test initializer 4" << endl;
    //input
    
    //conv
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[1])->bias = {0, 0};
    
    //fc
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[3])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize7(){
    cout << "[prompt]: Using test initializer 7" << endl;
    //input
    
    //conv1
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[1])->bias = {0, 0};
    
    //conv2
    ((ConvLayer*)net[3])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[3])->bias = {0, 0};

    //fc
    ((FcLayer*)net[5])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[5])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize8(){
    cout << "[prompt]: Using test initializer 8" << endl;
    //input
    
    //conv
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3})};
    ((ConvLayer*)net[1])->bias = {0};
    
    //fc
    ((FcLayer*)net[4])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[4])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize9(){
    cout << "[prompt]: Using test initializer 9" << endl;
    //input
    
    //conv1
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[1])->bias = {0.5, -0.5};
    
    //conv2
    ((ConvLayer*)net[3])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[3])->bias = {0.5, -0.5};
    
    //fc
    ((FcLayer*)net[5])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[5])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize10(){
    cout << "[prompt]: Using test initializer 10" << endl;
    //input
    
    //fc
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[3])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize11(){
    cout << "[prompt]: Using test initializer 11" << endl;
    //input
    
    //fc
    ((FcLayer*)net[1])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[1])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize12(){
    cout << "[prompt]: Using test initializer 12" << endl;
    //input
    
    //conv1
    ((ConvLayer*)net[1])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[1])->bias = {0, 0};
    
    //conv2
    ((ConvLayer*)net[2])->kernel = {vector<float>({0.1, 0.2, 0.3})};
    ((ConvLayer*)net[2])->bias = {0};
    
    //fc
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[3])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize13(){
    cout << "[prompt]: Using test initializer 13" << endl;
    //input
    
    //fc1
    ((FcLayer*)net[1])->weight = {vector<float>({0.1, 0.1}), vector<float>({0.2, 0.2}), vector<float>({0.3, 0.3}), vector<float>({0.4, 0.4})};
    ((FcLayer*)net[1])->bias = {0.1, 0.2, 0.3, 0.4};
    
    //fc2
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[3])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::testInitialize14(){
    cout << "[prompt]: Using test initializer 14" << endl;
    //input
    
    //fc1
    ((FcLayer*)net[1])->weight = {vector<float>({0.1, 0.1}), vector<float>({0.2, 0.2}), vector<float>({0, 0})};
    ((FcLayer*)net[1])->bias = {0.1, 0.2, 0};
    
    //conv
    ((ConvLayer*)net[2])->kernel = {vector<float>({0.1, 0.2, 0.3}), vector<float>({0.3, 0.2, 0.1})};
    ((ConvLayer*)net[2])->bias = {0, 0};
    
    //fc2
    ((FcLayer*)net[3])->weight = {vector<float>({0.1, 0.2}), vector<float>({-0.1, -0.2})};
    ((FcLayer*)net[3])->bias = {0.3, -0.3};
    
    mode = TEST;
}

void Model::paramCheck(){
    startLayerNo = (int)net.size();
    assert(net.front()->layerType == Layer::INPUT);
    assert(net.back()->layerType == Layer::FC);
    ((FcLayer*)net.back())->isOutputLayer = true;
    net.back()->next = nullptr;
    int layerNo = 0;
    cout << "   {Net structure}" << endl;
    while (layerNo < net.size()){
        switch (net[layerNo]->layerType) {
            case Layer::layerSet::INPUT:
                assert(layerNo == 0);
                cout << "\t(" << layerNo + 1 << ")\t" << "input: " << net[layerNo]->nodeNum << endl;
                break;
            case Layer::layerSet::CONV:
                startLayerNo = min(startLayerNo, layerNo);
                ((ConvLayer*)net[layerNo])->nodeNumComputation();
                cout << "\t(" << layerNo + 1 << ")\t" << "conv: " << net[layerNo]->nodeNum << endl;
                break;
            case Layer::layerSet::POOL:
                ((PoolLayer*)net[layerNo])->nodeNumComputation();
                cout << "\t(" << layerNo + 1 << ")\t" << "pool: " << net[layerNo]->nodeNum << endl;
                break;
            case Layer::layerSet::FC:
                startLayerNo = min(startLayerNo, layerNo);
                cout << "\t(" << layerNo + 1 << ")\t" << "fc: " << net[layerNo]->nodeNum << endl;
                break;
            default:
                cout << "[Error@Model:initialize]: Unable to handle layer type " << net[layerNo]->layerType << endl;
                exit(4);
                break;
        }
        layerNo++;
    }
    if (startLayerNo != 1){
        cout << "[Warning@Model:initialize]: Back propagation only reaches layer " << startLayerNo + 1 << endl;
    }
    cout << endl;
}

void Model::initialize(){
    for (int i = startLayerNo; i < net.size(); i++){
        switch (net[i]->layerType) {
            case Layer::layerSet::INPUT: case Layer::layerSet::POOL:
                break;
            case Layer::layerSet::CONV:
                ((ConvLayer*)net[i])->initialize();
                break;
            case Layer::layerSet::FC:
                ((FcLayer*)net[i])->initialize();
                break;
            default:
                cout << "[Error@Model:initialize]: Unable to handle layer type " << net[i]->layerType << endl;
                exit(4);
                break;
        }
    }
}

void Model::forwardPropagation(){
    int layerNo = (currentIteration == 0)? 1: startLayerNo;
    while (layerNo < net.size()){
        switch (net[layerNo]->layerType) {
            case Layer::layerSet::CONV:
                ((ConvLayer*)net[layerNo])->forward();
                break;
            case Layer::layerSet::POOL:
                ((PoolLayer*)net[layerNo])->forward();
                break;
            case Layer::layerSet::FC:
                ((FcLayer*)net[layerNo])->forward();
                break;
            default:
                cout << "[Error@Model:forwardPropagation]: Unable to handle layer type " << net[layerNo]->layerType << endl;
                exit(4);
                break;
        }
        layerNo++;
    }
}

void Model::crossEntropyLoss(){
    cout << "Out: ";
    for_each(net.back()->out.begin(), net.back()->out.end(), [](float ele){cout << " " << ele;});
    cout << endl << "Loss: ";
    cout << inner_product(output.begin(), output.end(), net.back()->out.begin(), 0.f, plus<float>(), [](float t, float y){return -t * log(y);});
    cout << endl;
}

void Model::squaredLoss(){
    cout << "Out: ";
    for_each(net.back()->out.begin(), net.back()->out.end(), [](float ele){cout << " " << ele;});
    cout << endl << "Loss: ";
    cout << 0.5 * inner_product(output.begin(), output.end(), net.back()->out.begin(), 0.f, plus<float>(), [](float t, float y){return pow(y - t, 2);});
    cout << endl;
}

void Model::logCoshLoss(){
    cout << "Out: ";
    for_each(net.back()->out.begin(), net.back()->out.end(), [](float ele){cout << " " << ele;});
    cout << endl << "Loss: ";
    cout << 0.5 * inner_product(output.begin(), output.end(), net.back()->out.begin(), 0.f, plus<float>(), [](float t, float y){return log(cosh(y - t));});
    cout << endl;
}

void Model::lossComputation(){
    cout << "Iteration(" << currentIteration + 1 << "/" << maxIteration << ")" << endl;
    switch (lossFunction) {
        case FcLayer::lossSet::cross_entropy:
            crossEntropyLoss();
            break;
        case FcLayer::lossSet::squared:
            squaredLoss();
            break;
        case FcLayer::lossSet::log_cosh:
            logCoshLoss();
            break;
        default:
            cout << "[Error@Model:lossComputation]: Unable to handle loss function " << lossFunction << endl;
            exit(4);
            break;
    }
    cout << endl;
}

void Model::backwardPropagation(){
    int layerNo = (int)net.size() - 2;
    ((FcLayer*)net.back())->backward(learningRate, output, lossFunction);
    
    while (layerNo >= startLayerNo){
        switch (net[layerNo]->layerType) {
            case Layer::layerSet::CONV:
                ((ConvLayer*)net[layerNo])->backward(learningRate);
                break;
            case Layer::layerSet::POOL:
                ((PoolLayer*)net[layerNo])->backward();
                break;
            case Layer::layerSet::FC:
                ((FcLayer*)net[layerNo])->backward(learningRate);
                break;
            default:
                cout << "[Error@Model:backwardPropagation]: Unable to recognize layer type " << net[layerNo]->layerType << endl;
                exit(4);
                break;
        }
        layerNo--;
    }
}

void Model::train(const vector<float>& in, const vector<float>& out, const float learn, const int max, const FcLayer::lossSet& loss){
    
    currentIteration = 0;
    lossFunction = loss;
    assert(learn > 0);
    learningRate = learn;
    maxIteration = max;
    output = out;
    net.front()->out = in;
    
    assert(net.front()->nodeNum == in.size());
    assert(net.back()->nodeNum == out.size());
    
    paramCheck();
    
    if (mode == NORMAL)
        initialize();
    
    while (currentIteration < maxIteration){
        forwardPropagation();
        lossComputation();
        backwardPropagation();
        currentIteration++;
    }
}

Model::~Model(){
    for (int i = 0; i < net.size(); i++){
        switch (net[i]->layerType) {
            case Layer::layerSet::INPUT:
                delete net[i];
                break;
            case Layer::layerSet::CONV:
                delete ((ConvLayer*)net[i]);
                break;
            case Layer::layerSet::POOL:
                delete ((PoolLayer*)net[i]);
                break;
            case Layer::layerSet::FC:
                delete ((FcLayer*)net[i]);
                break;
            default:
                cout << "[Error]: Unable to recognize layer type " << net[i]->layerType << endl;
                exit(4);
                break;
        }
    }
}
