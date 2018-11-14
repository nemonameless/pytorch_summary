# PyTorch_summary
Pytorch summary:

Calculate parameters quantity, memory, and flops in simple and complex way.

## simplesum mode:

```
  + Number of params: 25.56M
  + Number of FLOPs: 4.11G
```

## complexsum mode:

following  https://github.com/ceykmc/pytorch_model_summary , and add the support to  bilinear layer.

```
                 module name   input shape  output shape  parameter quantity inference memory(MB)         MAdd duration percent
0               conv1_Conv2d     3 224 224    64 112 112                9408               3.06MB  235,225,088            6.32%
1            bn1_BatchNorm2d    64 112 112    64 112 112                 128               3.06MB    3,211,264            0.91%
2                  relu_ReLU    64 112 112    64 112 112                   0               3.06MB      802,816            0.94%
3          maxpool_MaxPool2d    64 112 112    64  56  56                   0               0.77MB    1,605,632            3.58%
4      layer1.0.conv1_Conv2d    64  56  56    64  56  56                4096               0.77MB   25,489,408            0.45%
5   layer1.0.bn1_BatchNorm2d    64  56  56    64  56  56                 128               0.77MB      802,816            0.21%
6      layer1.0.conv2_Conv2d    64  56  56    64  56  56               36864               0.77MB  231,010,304            2.75%
7   layer1.0.bn2_BatchNorm2d    64  56  56    64  56  56                 128               0.77MB      802,816            0.22%
8      layer1.0.conv3_Conv2d    64  56  56   256  56  56               16384               3.06MB  101,957,632            1.38%
9   layer1.0.bn3_BatchNorm2d   256  56  56   256  56  56                 512               3.06MB    3,211,264            0.78%
10        layer1.0.relu_ReLU   256  56  56   256  56  56                   0               3.06MB      802,816            0.53%
11       layer1.0.downsample    64  56  56   256  56  56               16896               6.12MB  105,168,896            2.13%
12                  layer1.1   256  56  56   256  56  56               70400              12.25MB  441,147,392            6.28%
13                  layer1.2   256  56  56   256  56  56               70400              12.25MB  441,147,392            6.24%
14     layer2.0.conv1_Conv2d   256  56  56   128  56  56               32768               1.53MB  205,119,488            1.94%
15  layer2.0.bn1_BatchNorm2d   128  56  56   128  56  56                 256               1.53MB    1,605,632            0.20%
16     layer2.0.conv2_Conv2d   128  56  56   128  28  28              147456               0.38MB  231,110,656            2.65%
17  layer2.0.bn2_BatchNorm2d   128  28  28   128  28  28                 256               0.38MB      401,408            0.08%
18     layer2.0.conv3_Conv2d   128  28  28   512  28  28               65536               1.53MB  102,359,040            0.96%
19  layer2.0.bn3_BatchNorm2d   512  28  28   512  28  28                1024               1.53MB    1,605,632            0.25%
20        layer2.0.relu_ReLU   512  28  28   512  28  28                   0               1.53MB      401,408            0.26%
21       layer2.0.downsample   256  56  56   512  28  28              132096               3.06MB  206,725,120            1.93%
22                  layer2.1   512  28  28   512  28  28              280064               6.12MB  438,939,648            4.36%
23                  layer2.2   512  28  28   512  28  28              280064               6.12MB  438,939,648            4.47%
24                  layer2.3   512  28  28   512  28  28              280064               6.12MB  438,939,648            4.44%
25     layer3.0.conv1_Conv2d   512  28  28   256  28  28              131072               0.77MB  205,320,192            1.48%
26  layer3.0.bn1_BatchNorm2d   256  28  28   256  28  28                 512               0.77MB      802,816            0.14%
27     layer3.0.conv2_Conv2d   256  28  28   256  14  14              589824               0.19MB  231,160,832            2.02%
28  layer3.0.bn2_BatchNorm2d   256  14  14   256  14  14                 512               0.19MB      200,704            0.08%
29     layer3.0.conv3_Conv2d   256  14  14  1024  14  14              262144               0.77MB  102,559,744            0.77%
30  layer3.0.bn3_BatchNorm2d  1024  14  14  1024  14  14                2048               0.77MB      802,816            0.28%
31        layer3.0.relu_ReLU  1024  14  14  1024  14  14                   0               0.77MB      200,704            0.15%
32       layer3.0.downsample   512  28  28  1024  14  14              526336               1.53MB  206,123,008            1.76%
33                  layer3.1  1024  14  14  1024  14  14             1117184               3.06MB  437,835,776            3.97%
34                  layer3.2  1024  14  14  1024  14  14             1117184               3.06MB  437,835,776            4.06%
35                  layer3.3  1024  14  14  1024  14  14             1117184               3.06MB  437,835,776            4.05%
36                  layer3.4  1024  14  14  1024  14  14             1117184               3.06MB  437,835,776            4.03%
37                  layer3.5  1024  14  14  1024  14  14             1117184               3.06MB  437,835,776            4.02%
38     layer4.0.conv1_Conv2d  1024  14  14   512  14  14              524288               0.38MB  205,420,544            1.42%
39  layer4.0.bn1_BatchNorm2d   512  14  14   512  14  14                1024               0.38MB      401,408            0.14%
40     layer4.0.conv2_Conv2d   512  14  14   512   7   7             2359296               0.10MB  231,185,920            2.41%
41  layer4.0.bn2_BatchNorm2d   512   7   7   512   7   7                1024               0.10MB      100,352            0.13%
42     layer4.0.conv3_Conv2d   512   7   7  2048   7   7             1048576               0.38MB  102,660,096            1.03%
43  layer4.0.bn3_BatchNorm2d  2048   7   7  2048   7   7                4096               0.38MB      401,408            0.44%
44        layer4.0.relu_ReLU  2048   7   7  2048   7   7                   0               0.38MB      100,352            0.08%
45       layer4.0.downsample  1024  14  14  2048   7   7             2101248               0.77MB  205,821,952            2.48%
46                  layer4.1  2048   7   7  2048   7   7             4462592               1.53MB  437,283,840            5.26%
47                  layer4.2  2048   7   7  2048   7   7             4462592               1.53MB  437,283,840            5.27%
48         avgpool_AvgPool2d  2048   7   7  2048   1   1                   0               0.01MB      100,352            0.05%
49                 fc_Linear          2048          1000             2049000               0.00MB    4,095,000            0.24%
===============================================================================================================================
total parameters quantity: 25,557,032
total memory: 109.69MB
total MAdd: 8,219,737,624

```
## validation mode
 To make sure the result is correct, a validation mothod from https://github.com/sksq96/pytorch-summary is added.
 To use this validation method, please install  *torchsummary*
 `pip3 install torchsummary`
