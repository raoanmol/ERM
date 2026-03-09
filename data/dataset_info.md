# Dataset Information

Statistics computed with `seed=42` and `num_val_samples_per_class=0` (no validation carve-out).

## Summary

| Dataset | Classes | Train | Test | Total |
|---------|--------:|------:|-----:|------:|
| BAR | 6 | 1,941 | 654 | 2,595 |
| CelebA (attr: Male) | 2 | 172,703 | 29,896 | 202,599 |
| NICO++ | 60 | 66,649 | 22,217 | 88,866 |

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

## NICO++

60 classes across 6 visual contexts (autumn, dim, grass, outdoor, rock, water). All images reside in a single directory; train/test is a seeded 75/25 random split.

| Class | ID | Train | Test | Total |
|-------|---:|------:|-----:|------:|
| car | 0 | 1,357 | 463 | 1,820 |
| flower | 1 | 1,322 | 467 | 1,789 |
| chair | 2 | 1,201 | 376 | 1,577 |
| truck | 3 | 1,335 | 449 | 1,784 |
| tiger | 4 | 1,135 | 399 | 1,534 |
| wheat | 5 | 669 | 253 | 922 |
| seal | 6 | 1,062 | 330 | 1,392 |
| wolf | 7 | 1,134 | 386 | 1,520 |
| lion | 8 | 1,284 | 407 | 1,691 |
| dolphin | 9 | 778 | 249 | 1,027 |
| lifeboat | 10 | 966 | 342 | 1,308 |
| corn | 11 | 694 | 214 | 908 |
| fishing rod | 12 | 1,355 | 443 | 1,798 |
| owl | 13 | 1,054 | 357 | 1,411 |
| sunflower | 14 | 902 | 325 | 1,227 |
| cow | 15 | 1,618 | 544 | 2,162 |
| bird | 16 | 1,711 | 588 | 2,299 |
| clock | 17 | 896 | 297 | 1,193 |
| shrimp | 18 | 427 | 133 | 560 |
| goose | 19 | 1,093 | 351 | 1,444 |
| airplane | 20 | 1,156 | 342 | 1,498 |
| rabbit | 21 | 868 | 296 | 1,164 |
| hot air balloon | 22 | 1,311 | 414 | 1,725 |
| lizard | 23 | 1,234 | 386 | 1,620 |
| hat | 24 | 1,040 | 359 | 1,399 |
| spider | 25 | 830 | 281 | 1,111 |
| motorcycle | 26 | 1,451 | 491 | 1,942 |
| tortoise | 27 | 1,102 | 395 | 1,497 |
| dog | 28 | 1,996 | 707 | 2,703 |
| crocodile | 29 | 939 | 275 | 1,214 |
| elephant | 30 | 1,290 | 417 | 1,707 |
| gun | 31 | 914 | 290 | 1,204 |
| fox | 32 | 916 | 334 | 1,250 |
| bus | 33 | 1,195 | 403 | 1,598 |
| cat | 34 | 1,481 | 510 | 1,991 |
| sailboat | 35 | 1,333 | 457 | 1,790 |
| giraffe | 36 | 1,187 | 386 | 1,573 |
| cactus | 37 | 1,040 | 339 | 1,379 |
| pumpkin | 38 | 946 | 317 | 1,263 |
| train | 39 | 1,148 | 371 | 1,519 |
| ship | 40 | 1,281 | 438 | 1,719 |
| helicopter | 41 | 1,224 | 409 | 1,633 |
| bicycle | 42 | 1,509 | 515 | 2,024 |
| racket | 43 | 614 | 216 | 830 |
| squirrel | 44 | 1,175 | 357 | 1,532 |
| bear | 45 | 1,371 | 477 | 1,848 |
| scooter | 46 | 739 | 253 | 992 |
| mailbox | 47 | 757 | 274 | 1,031 |
| horse | 48 | 1,375 | 474 | 1,849 |
| pineapple | 49 | 673 | 212 | 885 |
| frog | 50 | 1,028 | 330 | 1,358 |
| football | 51 | 756 | 206 | 962 |
| ostrich | 52 | 888 | 288 | 1,176 |
| tent | 53 | 1,425 | 488 | 1,913 |
| kangaroo | 54 | 912 | 326 | 1,238 |
| monkey | 55 | 1,092 | 369 | 1,461 |
| crab | 56 | 691 | 219 | 910 |
| sheep | 57 | 1,548 | 488 | 2,036 |
| butterfly | 58 | 984 | 332 | 1,316 |
| umbrella | 59 | 1,237 | 403 | 1,640 |
| **Total** | | **66,649** | **22,217** | **88,866** |
