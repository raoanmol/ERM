# Dataset Information

Statistics computed with `seed=42` and `num_val_samples_per_class=0` (no validation carve-out).

## Summary

| Dataset | Classes | Train | Test | Total |
|---------|--------:|------:|-----:|------:|
| BAR | 6 | 1,941 | 654 | 2,595 |
| CelebA (attr: Male) | 2 | 172,703 | 29,896 | 202,599 |
| NICO++ (DG_Benchmark) | 60 | 71,091 | 17,775 | 88,866 |

## BAR

6 classes. Train and test come from separate directories (`train/`, `test/`).

| Class | Train | Test | Total |
|-------|------:|-----:|------:|
| climbing | 326 | 105 | 431 |
| diving | 520 | 159 | 679 |
| fishing | 163 | 42 | 205 |
| pole vaulting | 279 | 131 | 410 |
| racing | 336 | 132 | 468 |
| throwing | 317 | 85 | 402 |
| **Total** | **1,941** | **654** | **2,595** |

## CelebA

2 classes (binary attribute: `Male`). Official partition 0 = train, partition 2 = test, partition 1 (validation) is split 50/50 into train and test.

| Class | Train | Test | Total |
|-------|------:|-----:|------:|
| Not Male (0) | 100,157 | 18,008 | 118,165 |
| Male (1) | 72,546 | 11,888 | 84,434 |
| **Total** | **172,703** | **29,896** | **202,599** |

## NICO++ (DG_Benchmark)

60 classes across 6 visual contexts (autumn, dim, grass, outdoor, rock, water). Train/test split is defined by official DG_Benchmark annotation files (80/20 split).

### Class Distribution

| Class | ID | Train | Test | Total |
|-------|---:|------:|-----:|------:|
| airplane | 20 | 1,198 | 300 | 1,498 |
| bear | 45 | 1,471 | 377 | 1,848 |
| bicycle | 42 | 1,623 | 401 | 2,024 |
| bird | 16 | 1,817 | 482 | 2,299 |
| bus | 33 | 1,260 | 338 | 1,598 |
| butterfly | 58 | 1,053 | 263 | 1,316 |
| cactus | 37 | 1,118 | 261 | 1,379 |
| car | 0 | 1,480 | 340 | 1,820 |
| cat | 34 | 1,628 | 363 | 1,991 |
| chair | 2 | 1,237 | 340 | 1,577 |
| clock | 17 | 939 | 254 | 1,193 |
| corn | 11 | 721 | 187 | 908 |
| cow | 15 | 1,725 | 437 | 2,162 |
| crab | 56 | 743 | 167 | 910 |
| crocodile | 29 | 987 | 227 | 1,214 |
| dog | 28 | 2,177 | 526 | 2,703 |
| dolphin | 9 | 822 | 205 | 1,027 |
| elephant | 30 | 1,360 | 347 | 1,707 |
| fishing rod | 12 | 1,442 | 356 | 1,798 |
| flower | 1 | 1,413 | 376 | 1,789 |
| football | 51 | 776 | 186 | 962 |
| fox | 32 | 1,002 | 248 | 1,250 |
| frog | 50 | 1,078 | 280 | 1,358 |
| giraffe | 36 | 1,253 | 320 | 1,573 |
| goose | 19 | 1,155 | 289 | 1,444 |
| gun | 31 | 991 | 213 | 1,204 |
| hat | 24 | 1,132 | 267 | 1,399 |
| helicopter | 41 | 1,323 | 310 | 1,633 |
| horse | 48 | 1,497 | 352 | 1,849 |
| hot air balloon | 22 | 1,363 | 362 | 1,725 |
| kangaroo | 54 | 998 | 240 | 1,238 |
| lifeboat | 10 | 1,041 | 267 | 1,308 |
| lion | 8 | 1,339 | 352 | 1,691 |
| lizard | 23 | 1,288 | 332 | 1,620 |
| mailbox | 47 | 815 | 216 | 1,031 |
| monkey | 55 | 1,157 | 304 | 1,461 |
| motorcycle | 26 | 1,590 | 352 | 1,942 |
| ostrich | 52 | 932 | 244 | 1,176 |
| owl | 13 | 1,113 | 298 | 1,411 |
| pineapple | 49 | 694 | 191 | 885 |
| pumpkin | 38 | 1,028 | 235 | 1,263 |
| rabbit | 21 | 924 | 240 | 1,164 |
| racket | 43 | 671 | 159 | 830 |
| sailboat | 35 | 1,436 | 354 | 1,790 |
| scooter | 46 | 776 | 216 | 992 |
| seal | 6 | 1,109 | 283 | 1,392 |
| sheep | 57 | 1,620 | 416 | 2,036 |
| ship | 40 | 1,388 | 331 | 1,719 |
| shrimp | 18 | 455 | 105 | 560 |
| spider | 25 | 865 | 246 | 1,111 |
| squirrel | 44 | 1,212 | 320 | 1,532 |
| sunflower | 14 | 983 | 244 | 1,227 |
| tent | 53 | 1,539 | 374 | 1,913 |
| tiger | 4 | 1,216 | 318 | 1,534 |
| tortoise | 27 | 1,213 | 284 | 1,497 |
| train | 39 | 1,215 | 304 | 1,519 |
| truck | 3 | 1,433 | 351 | 1,784 |
| umbrella | 59 | 1,315 | 325 | 1,640 |
| wheat | 5 | 734 | 188 | 922 |
| wolf | 7 | 1,208 | 312 | 1,520 |
| **Total** | | **71,091** | **17,775** | **88,866** |

### Context Distribution

| Context | Train | Test | Total |
|---------|------:|-----:|------:|
| autumn | 7,275 | 1,819 | 9,094 |
| dim | 9,991 | 2,498 | 12,489 |
| grass | 16,261 | 4,066 | 20,327 |
| outdoor | 13,352 | 3,338 | 16,690 |
| rock | 8,851 | 2,213 | 11,064 |
| water | 15,361 | 3,841 | 19,202 |
| **Total** | **71,091** | **17,775** | **88,866** |
