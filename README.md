![Medieval coin similarity processing](coinsim.webp)


# Medieval coin comparison

Similarity measures for comparing normal maps of medieval coins previously
acquired using photometric stereo analysis [1].


## Usage

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

[1] https://fordatis.fraunhofer.de/bitstream/fordatis/281/


## License

This software is licensed under the MIT license. See LICENSE.txt for details.

