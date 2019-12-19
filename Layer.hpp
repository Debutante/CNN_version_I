//
//  Layer.hpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include <vector>

using namespace std;

class Layer{
public:
    enum layerSet {INPUT, CONV, POOL, FC};
    Layer* prev;
    Layer* next;
    int nodeNum;
    layerSet layerType;
    vector<float> out;
    
    Layer(Layer*);
    Layer(const int, Layer*);
};

#endif /* Layer_hpp */
