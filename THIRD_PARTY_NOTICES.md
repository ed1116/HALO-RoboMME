# Third-party notices

This repository preserves and extends the official HALO source at commit
`7778638401cc07043e59e2381e84f2383401f491`, licensed under Apache-2.0 in
[`LICENSE.txt`](LICENSE.txt).

The RoboMME adaptation interoperates with, but does not vendor:

- [RoboMME policy learning](https://github.com/RoboMME/robomme_policy_learning), Apache-2.0, reference commit `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`.
- [RoboMME benchmark](https://github.com/RoboMME/robomme_benchmark), reference commit `856bc3a189d4172f3f47dbee4424d585f8d78db3`.
- [CrossMAE / ICRT weights](https://huggingface.co/mlfu7/ICRT), downloaded separately and governed by the model repository's terms.
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), Apache-2.0, used offline to generate and judge VQA supervision.

Datasets and model weights are not redistributed by this repository. Their exact
revisions and checksums are recorded in run manifests when acquired.

