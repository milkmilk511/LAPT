# LAPT
Codes for Design Principle Transfer in Neural Architecture Search via Large Language Models. This work has been accepted by AAAI.
Now, we open the codes for NAS-bench-201 and Trans-bench-101. Codes for DARTs will be accessed later.

# Settings
Running these experiments on benchmarks is needed for GPU. You can try the proposed LATP easily. However, you should install the OpenAI parkage and guarantee the available use of LLMs such as GPT.
We recomment the use of GPT-4o and Gemini which are faster.

GPT-4 can achieve the better performance, but it is time-consuming and unstable

# NAS-bench-201
You can download the file to reproduce the MTNAS method on the NAS benchmark 201. You can download the NAS201 dataset by yourself from https://github.com/D-X-Y/NAS-Bench-201 and place it in the same file with the .py file. Also, three .Jason files are provided by use and these files include the architectural information to support the experiment.

# Trans-bench-101
You can download the file to reproduce the MTNAS method on the NAS benchmark 201. You can download the NAS201 dataset by yourself from https://github.com/yawen-d/TransNASBench and place it in the same file with the .py file.

# Citation
@article{zhou2024design,
  title={Design Principle Transfer in Neural Architecture Search via Large Language Models},
  author={Zhou, Xun and Feng, Liang and Wu, Xingyu and Lu, Zhichao and Tan, Kay Chen},
  journal={arXiv preprint arXiv:2408.11330},
  year={2024}
}
