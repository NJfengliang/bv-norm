# BV-NORM: A neural operator learning framework for parametric boundary value problems on complex geometric domains in engineering
- This repository contains codes and datas accompanying our paper : _BV-NORM: A neural operator learning framework for parametric boundary value problems on complex geometric domains in engineering_. 
- For more details, feel free to contact us (fengliang@nuaa.edu.cn).
  
## BV-NORM
- ![images](image/framework1.png)
- ![images](image/framework2.png)
  
## Dependencies & packages
Dependencies:
- Python (tested on 3.8.13)
- PyTorch (tested on 1.11.0)
- lapy (tested on 0.6.0)

## Data
The shapes of dataset about the Case1-Case4:
```
Case1-mult_holes.mat
├── Input: f_bc  2000*101 (BC-NET) MeshNodes 2*1119 (Geo-NET)
└── Output: u_field   2000*1119
---------------------------------------------
Case2-data_heat_3.mat
├── Input: input  300*186 (BC-NET) output_points 7199*3 (Geo-NET)
└── Output: U_source  300*7199
---------------------------------------------
Case3-Qianyuan_T2.mat
├── Input: Tair_time  600*151*2 (BC-NET) nodes 2743*3 (Geo-NET)
└── Output: D_field 600*2743
---------------------------------------------
Case4-BloodFlow.mat
├── Input: BC_time  500*121*6 (BC-NET) nodes 1656*3 (Geo-NET)
└── Output: `velocity_x`(500*1656),`velocity_y`(500*1656),`velocity_z`(500*1656)
```

## Results
### Case1
- ![images](image/Case1.png)
---------------------------------------------------
### Case2
- ![images](image/Case2.png)
---------------------------------------------------
### Case3
- ![images](image/Case3.png)
---------------------------------------------------
### Case4
- ![images](image/Case4.png)
---------------------------------------------------




