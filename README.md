![Medieval coin similarity processing](coinsim.webp)


# Medieval coin comparison

Similarity measures for comparing normal maps of medieval coins previously
acquired using photometric stereo analysis [1,2].


## Overview

An overview of the coin processing pipeline is presented in the Figure below.
Data is flowing from left to right. Following a preprocessing step, coins are
spatially aligned before various global and local similarity measures are
computed. Finally, the results are analysed by their similarity distributions
and using a coin classification task.

```mermaid
flowchart LR
    
    subgraph Preprocessing
    A[RGB \n decoding] --> X[Downscaling]
    X --> B[Bending \n correction]
    end

    subgraph Spatial alignment
    B --> D[Coin rim detection]
    D --> E[Translational \n alignment]
    E --> F[Rotational \n alignment]
    end

    subgraph Global similarity
    B --> C[Gradient field \n 2D histogram]
    F --> G[Normal field \n similarity]
    end

    subgraph Analysis
    G --> I[Coin classification]
    C --> I
    C --> J
    G --> J[Similarity distribution]
    D --> K[Rim detection validation]
    end

    subgraph Local similarity
    G --> H[Sliding window similarity]
    end
```


## Usage

A Linux system with Python and Conda is needed to run the coin processing
pipeline.

1. Create and activate a new conda environment
```
conda env create -f environment.yml
conda activate coins
```

2. Run the whole pipeline
```
bash run.sh
```

3. Intermediate processing output are stored in directory `output/`, and
resulting data and figures from analysis are saved in directory `results/`.


## References

[1] Original data publication (https://doi.org/10.5334/joad.116)\
[2] Original data repository (https://doi.org/10.24406/fordatis/210)


## License

This software is licensed under the MIT license. See LICENSE.txt for details.

