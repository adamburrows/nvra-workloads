# PyTorch MNIST Example

This document introduces **PyTorch** and the **MNIST dataset**, and demonstrates how to run a distributed training job on a Kubernetes cluster using **NVIDIA Run:ai**.

---

## Introduction

**PyTorch** is a popular open-source machine learning framework for building deep learning models.  
**MNIST** is a classic dataset of handwritten digits (0-9), commonly used for benchmarking image classification algorithms.

This example is designed to work with the **Fake GPU operator** for testing purposes, so no actual GPU hardware is required.

---

## Container Image

We will use a prebuilt container image.


<details>
<summary>Interested in building your own image?</summary>

1. Download the files in the build-image directory.
2. Optionally make modifications to the Dockerfile and/or .py file.
3. Build, tag, and push the image.
```
podman build -t my-test-image:latest --platform linux/amd64 .
podman tag localhost/my-test-image:latest docker.io/<user>/my-test-image:latest
podman push docker.io/<user>/my-test-image:latest
```
</details>

Run the PyTorch workload.  

## Runai Command - PT workload
```bash
runai pytorch submit \
-p testproject \
-i docker.io/hireamb/pytorch-mnist-example:latest \
-g 1 \
--workers=2 \
--working-dir /opt/pytorch-mnist \
--existing-pvc claimname=dist-datasets-v3-project-hpzdp,path=/katib \
--command -- python3 "/opt/pytorch-mnist/mnist.py" "--epochs=1"
```
| Flag            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `-p`            | Run:ai project                                                              |
| `-i`            | Docker/Podman image to use                                                  |
| `-g`            | Number of whole GPUs                                                        |
| `--workers`     | Number of worker pods (2 in this case; 1 master pod is implied)             |
| `--working-dir` | Default directory when entering the container                               |
| `--existing-pvc`| Mount persistent storage so we can save the accuracy/loss metrics                       |
| `--command`     | Overrides the container's entrypoint; the command after `--` is executed    |

Copy the plot-metrics.py code into the notebook after launching this workspace to view training metrics such as accuracy and loss.  

## Runai Command - Jupyter Notebook
```bash
runai workspace submit \
-p testproject \
-i jupyter/scipy-notebook \
-g 1 \
--external-url container=8888 \
--existing-pvc claimname=dist-datasets-v3-project-hpzdp,path=/opt/data \
--command -- start-notebook.sh --NotebookApp.base_url=/\${RUNAI_PROJECT}/\${RUNAI_JOB_NAME} --NotebookApp.token=''
```

| Flag                     | Description                                                                    |
|--------------------------|--------------------------------------------------------------------------------|
| `-p`                     | Run:ai project                                                                 |
| `-i`                     | Docker/Podman image to use                                                     |
| `-g`  | Number of whole GPUs                                                         |
| `--external-url`         | Create an ingress connection available from the UI                             |
| `--existing-pvc`         | Mount persistent storage so we can save the accuracy/loss metrics              |
| `--command`              | Overrides the container's entrypoint; the command after `--` is executed       |