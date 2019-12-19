//
//  PoolLayer.hpp
//  test1203
//
//  Created by 许清嘉 on 12/5/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef PoolLayer_hpp
#define PoolLayer_hpp

#include "Layer.hpp"

class PoolLayer: public Layer{
public:
    int poolSize;
    enum poolSet {general, overlap};
    enum sampleSet {max, average};
    int poolStride;
    poolSet poolType;
    sampleSet sampleType;
    vector<vector<float>> weight;
    vector<float> coef;
    
    PoolLayer(const int, const poolSet&, const sampleSet&, Layer*);
    PoolLayer(const int, const poolSet&, const sampleSet&, Layer*, const int);
    void nodeNumComputation();
    void forward();
    void backward();
    
private:

    void coefFromConvFull();
    void coefFromConvSame();
    void coefFromConvValid();
    void maxPool();
    void avgPool();
    void coefFromFc();
    void coefFromPoolOverlap();
    void coefFromPoolGeneral();

};

#endif /* PoolLayer_hpp */
