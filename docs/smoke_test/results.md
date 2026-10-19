| arm | L1 seen ids | L1 unseen ids | 16px block ratio | output pixel std | AdaIN scale (body / out) |
|---|---|---|---|---|---|
| legacy (original code) | 0.241 | 0.235 | 3.74 | 0.092 | 0.020 / 0.225 |
| fixed, but raw AdaIN + old head | 0.137 | 0.209 | 1.39 | 0.153 | 0.059 / 0.432 |
| fixed | 0.116 | 0.202 | 1.02 | 0.194 | 0.981 / 1.006 |
| *ground truth* | – | – | 1.12 | – | – |
| *baseline: mean of K source frames* | 0.054 | 0.067 | | | |
| *baseline: dataset mean image* | 0.230 | 0.218 | | | |
