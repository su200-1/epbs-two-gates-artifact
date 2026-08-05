# Exact PTC × payment-quorum boundary

| Not-present PTC votes | Payment weight | Parent status | Payment | Builder debit (gwei) |
|---:|---:|:---:|:---:|---:|
| 255 | 59% | FULL | expired | 0 |
| 255 | 60% | FULL | settled | 100000000 |
| 255 | 61% | FULL | settled | 100000000 |
| 256 | 59% | FULL | expired | 0 |
| 256 | 60% | FULL | settled | 100000000 |
| 256 | 61% | FULL | settled | 100000000 |
| 257 | 59% | EMPTY | expired | 0 |
| 257 | 60% | EMPTY | settled | 100000000 |
| 257 | 61% | EMPTY | settled | 100000000 |
| 258 | 59% | EMPTY | expired | 0 |
| 258 | 60% | EMPTY | settled | 100000000 |
| 258 | 61% | EMPTY | settled | 100000000 |

结论：payload continuation 在 257 个 not-present votes 处由 FULL 跳变为 EMPTY；builder payment 在 60% regular-attestation weight 处结算。两条谓词相互独立。
