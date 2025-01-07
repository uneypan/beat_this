
```bash
python launch_scripts/compute_paper_metrics.py --models fold0  fold1 fold2 fold3 fold4 fold5 fold6 fold7 --datasplit val --aggregation-type k-fold
```

```bash
python launch_scripts/compute_paper_metrics.py --models fold0  fold1 fold2 fold3 fold4 fold5 fold6 fold7 --datasplit val --aggregation-type k-fold --num_workers=2 --postprocessor minimal 
```

'postprocessor' can be ["minimal", "dbn", 'bf', "dp", "sppk", 'plpdp', 'pf']

### Main 
```bash
python launch_scripts/compute_paper_metrics.py --models fold0  fold1 fold2 fold3 fold4 fold5 fold6 fold7 --datasplit val --aggregation-type k-fold  --postprocessor dbn --num_workers=2
```
Dataset metrics
F-measure_beat
asap: 0.763
ballroom: 0.975
beatles: 0.945
candombe: 0.997
filosax: 0.995
groove_midi: 0.937
guitarset: 0.92
hainsworth: 0.919
harmonix: 0.958
hjdb: 0.982
jaah: 0.951
rwc_classical: 0.77
rwc_jazz: 0.833
rwc_popular: 0.961
rwc_royalty-free: 0.945
simac: 0.779
smc: 0.627
tapcorrect: 0.93
------
Cemgil_beat
asap: 0.689
ballroom: 0.916
beatles: 0.846
candombe: 0.908
filosax: 0.923
groove_midi: 0.843
guitarset: 0.84
hainsworth: 0.867
harmonix: 0.866
hjdb: 0.952
jaah: 0.853
rwc_classical: 0.621
rwc_jazz: 0.713
rwc_popular: 0.907
rwc_royalty-free: 0.745
simac: 0.721
smc: 0.497
tapcorrect: 0.874
------
CMLt_beat
asap: 0.503
ballroom: 0.964
beatles: 0.872
candombe: 0.998
filosax: 0.988
groove_midi: 0.871
guitarset: 0.824
hainsworth: 0.84
harmonix: 0.899
hjdb: 0.972
jaah: 0.885
rwc_classical: 0.518
rwc_jazz: 0.724
rwc_popular: 0.901
rwc_royalty-free: 0.874
simac: 0.558
smc: 0.514
tapcorrect: 0.819
------
AMLt_beat
asap: 0.578
ballroom: 0.97
beatles: 0.93
candombe: 0.998
filosax: 0.988
groove_midi: 0.914
guitarset: 0.901
hainsworth: 0.909
harmonix: 0.94
hjdb: 0.979
jaah: 0.903
rwc_classical: 0.59
rwc_jazz: 0.751
rwc_popular: 0.936
rwc_royalty-free: 0.94
simac: 0.851
smc: 0.61
tapcorrect: 0.89
------
F-measure_downbeat
asap: 0.612
ballroom: 0.953
beatles: 0.888
candombe: 0.997
filosax: 0.985
groove_midi: 0.821
guitarset: 0.882
hainsworth: 0.8
harmonix: 0.907
hjdb: 0.966
jaah: 0.85
rwc_classical: 0.663
rwc_jazz: 0.807
rwc_popular: 0.937
rwc_royalty-free: 0.919
simac: 0.0
smc: 0.0
tapcorrect: 0.864
------
Cemgil_downbeat
asap: 0.58
ballroom: 0.901
beatles: 0.8
candombe: 0.905
filosax: 0.906
groove_midi: 0.763
guitarset: 0.807
hainsworth: 0.774
harmonix: 0.823
hjdb: 0.934
jaah: 0.768
rwc_classical: 0.547
rwc_jazz: 0.692
rwc_popular: 0.881
rwc_royalty-free: 0.716
simac: 0.0
smc: 0.0
tapcorrect: 0.816
------
CMLt_downbeat
asap: 0.235
ballroom: 0.93
beatles: 0.737
candombe: 0.999
filosax: 0.961
groove_midi: 0.727
guitarset: 0.798
hainsworth: 0.636
harmonix: 0.812
hjdb: 0.948
jaah: 0.715
rwc_classical: 0.342
rwc_jazz: 0.712
rwc_popular: 0.87
rwc_royalty-free: 0.822
simac: 0.0
smc: 0.0
tapcorrect: 0.682
------
AMLt_downbeat
asap: 0.439
ballroom: 0.941
beatles: 0.82
candombe: 0.999
filosax: 0.961
groove_midi: 0.819
guitarset: 0.874
hainsworth: 0.751
harmonix: 0.859
hjdb: 0.956
jaah: 0.742
rwc_classical: 0.476
rwc_jazz: 0.767
rwc_popular: 0.892
rwc_royalty-free: 0.822
simac: 0.0
smc: 0.0
tapcorrect: 0.765
------


### DBN
```bash
python launch_scripts/compute_paper_metrics.py --models fold0  fold1 fold2 fold3 fold4 fold5 fold6 fold7 --datasplit val --aggregation-type k-fold  --postprocessor dbn
```

Dataset metrics
F-measure_beat
asap: 0.715
ballroom: 0.969
beatles: 0.939
candombe: 0.996
filosax: 0.932
groove_midi: 0.929
guitarset: 0.93
hainsworth: 0.906
harmonix: 0.956
hjdb: 0.982
jaah: 0.847
rwc_classical: 0.726
rwc_jazz: 0.839
rwc_popular: 0.963
rwc_royalty-free: 0.946
simac: 0.775
smc: 0.575
tapcorrect: 0.919
------
Cemgil_beat
asap: 0.644
ballroom: 0.91
beatles: 0.839
candombe: 0.907
filosax: 0.861
groove_midi: 0.838
guitarset: 0.851
hainsworth: 0.853
harmonix: 0.862
hjdb: 0.952
jaah: 0.759
rwc_classical: 0.59
rwc_jazz: 0.719
rwc_popular: 0.908
rwc_royalty-free: 0.748
simac: 0.715
smc: 0.466
tapcorrect: 0.863
------
CMLt_beat
asap: 0.493
ballroom: 0.956
beatles: 0.873
candombe: 0.996
filosax: 0.855
groove_midi: 0.864
guitarset: 0.857
hainsworth: 0.841
harmonix: 0.904
hjdb: 0.976
jaah: 0.675
rwc_classical: 0.491
rwc_jazz: 0.755
rwc_popular: 0.917
rwc_royalty-free: 0.883
simac: 0.585
smc: 0.474
tapcorrect: 0.817
------
AMLt_beat
asap: 0.62
ballroom: 0.971
beatles: 0.93
candombe: 0.996
filosax: 0.864
groove_midi: 0.93
guitarset: 0.944
hainsworth: 0.907
harmonix: 0.947
hjdb: 0.983
jaah: 0.763
rwc_classical: 0.644
rwc_jazz: 0.816
rwc_popular: 0.96
rwc_royalty-free: 0.95
simac: 0.876
smc: 0.655
tapcorrect: 0.908
------
F-measure_downbeat
asap: 0.584
ballroom: 0.956
beatles: 0.88
candombe: 0.997
filosax: 0.931
groove_midi: 0.834
guitarset: 0.892
hainsworth: 0.805
harmonix: 0.913
hjdb: 0.966
jaah: 0.793
rwc_classical: 0.633
rwc_jazz: 0.736
rwc_popular: 0.939
rwc_royalty-free: 0.934
simac: 0.0
smc: 0.0
tapcorrect: 0.88
------
Cemgil_downbeat
asap: 0.55
ballroom: 0.903
beatles: 0.794
candombe: 0.905
filosax: 0.854
groove_midi: 0.777
guitarset: 0.819
hainsworth: 0.779
harmonix: 0.83
hjdb: 0.934
jaah: 0.717
rwc_classical: 0.521
rwc_jazz: 0.655
rwc_popular: 0.885
rwc_royalty-free: 0.738
simac: 0.0
smc: 0.0
tapcorrect: 0.827
------
CMLt_downbeat
asap: 0.389
ballroom: 0.953
beatles: 0.792
candombe: 0.999
filosax: 0.858
groove_midi: 0.788
guitarset: 0.817
hainsworth: 0.76
harmonix: 0.864
hjdb: 0.967
jaah: 0.649
rwc_classical: 0.4
rwc_jazz: 0.515
rwc_popular: 0.897
rwc_royalty-free: 0.87
simac: 0.0
smc: 0.0
tapcorrect: 0.791
------
AMLt_downbeat
asap: 0.606
ballroom: 0.969
beatles: 0.88
candombe: 0.999
filosax: 0.867
groove_midi: 0.895
guitarset: 0.906
hainsworth: 0.848
harmonix: 0.925
hjdb: 0.977
jaah: 0.751
rwc_classical: 0.585
rwc_jazz: 0.812
rwc_popular: 0.94
rwc_royalty-free: 0.937
simac: 0.0
smc: 0.0
tapcorrect: 0.877
------
```bash
python launch_scripts/compute_paper_metrics.py --models fold0  fold1 fold2 fold3 fold4 fold5 fold6 fold7 --datasplit val --aggregation-type k-fold --num_workers=2 --postprocessor bf 
```


Dataset metrics
F-measure_beat
asap: 0.711
ballroom: 0.886
beatles: 0.889
candombe: 0.954
filosax: 0.909
groove_midi: 0.914
guitarset: 0.903
hainsworth: 0.873
harmonix: 0.884
hjdb: 0.924
jaah: 0.794
rwc_classical: 0.707
rwc_jazz: 0.797
rwc_popular: 0.899
rwc_royalty-free: 0.849
simac: 0.764
smc: 0.594
tapcorrect: 0.856
------
Cemgil_beat
asap: 0.582
ballroom: 0.791
beatles: 0.759
candombe: 0.745
filosax: 0.767
groove_midi: 0.776
guitarset: 0.799
hainsworth: 0.767
harmonix: 0.724
hjdb: 0.861
jaah: 0.687
rwc_classical: 0.514
rwc_jazz: 0.616
rwc_popular: 0.771
rwc_royalty-free: 0.588
simac: 0.671
smc: 0.455
tapcorrect: 0.697
------
CMLt_beat
asap: 0.446
ballroom: 0.761
beatles: 0.776
candombe: 0.858
filosax: 0.757
groove_midi: 0.832
guitarset: 0.798
hainsworth: 0.752
harmonix: 0.72
hjdb: 0.795
jaah: 0.567
rwc_classical: 0.474
rwc_jazz: 0.728
rwc_popular: 0.766
rwc_royalty-free: 0.789
simac: 0.605
smc: 0.511
tapcorrect: 0.714
------
AMLt_beat
asap: 0.546
ballroom: 0.835
beatles: 0.836
candombe: 0.859
filosax: 0.768
groove_midi: 0.882
guitarset: 0.911
hainsworth: 0.86
harmonix: 0.839
hjdb: 0.918
jaah: 0.641
rwc_classical: 0.575
rwc_jazz: 0.77
rwc_popular: 0.871
rwc_royalty-free: 0.9
simac: 0.859
smc: 0.625
tapcorrect: 0.78
------
F-measure_downbeat
asap: 0.472
ballroom: 0.544
beatles: 0.489
candombe: 0.533
filosax: 0.684
groove_midi: 0.417
guitarset: 0.409
hainsworth: 0.48
harmonix: 0.461
hjdb: 0.635
jaah: 0.631
rwc_classical: 0.439
rwc_jazz: 0.478
rwc_popular: 0.448
rwc_royalty-free: 0.474
simac: 0.0
smc: 0.0
tapcorrect: 0.416
------
Cemgil_downbeat
asap: 0.426
ballroom: 0.561
beatles: 0.513
candombe: 0.55
filosax: 0.67
groove_midi: 0.453
guitarset: 0.45
hainsworth: 0.502
harmonix: 0.476
hjdb: 0.705
jaah: 0.592
rwc_classical: 0.356
rwc_jazz: 0.435
rwc_popular: 0.482
rwc_royalty-free: 0.414
simac: 0.0
smc: 0.0
tapcorrect: 0.429
------
CMLt_downbeat
asap: 0.124
ballroom: 0.132
beatles: 0.027
candombe: 0.0
filosax: 0.234
groove_midi: 0.006
guitarset: 0.005
hainsworth: 0.031
harmonix: 0.006
hjdb: 0.007
jaah: 0.311
rwc_classical: 0.118
rwc_jazz: 0.115
rwc_popular: 0.009
rwc_royalty-free: 0.0
simac: 0.0
smc: 0.0
tapcorrect: 0.017
------
AMLt_downbeat
asap: 0.329
ballroom: 0.415
beatles: 0.437
candombe: 0.557
filosax: 0.759
groove_midi: 0.217
guitarset: 0.23
hainsworth: 0.383
harmonix: 0.35
hjdb: 0.841
jaah: 0.619
rwc_classical: 0.266
rwc_jazz: 0.427
rwc_popular: 0.313
rwc_royalty-free: 0.444
simac: 0.0
smc: 0.0
tapcorrect: 0.241
------