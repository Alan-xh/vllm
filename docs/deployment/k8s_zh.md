---
title: 使用 Kubernetes
---
[](){ #deployment-k8s }

在 Kubernetes 上部署 vLLM 是一种可扩展且高效的方式来服务机器学习模型。本指南将引导您通过原生 Kubernetes 部署 vLLM。

* [使用 CPU 部署](#deployment-with-cpus)
* [使用 GPU 部署](#deployment-with-gpus)

或者，您可以使用以下任一方式将 vLLM 部署到 Kubernetes：

* [Helm](frameworks/helm.md)
* [InftyAI/llmaz](integrations/llmaz.md)
* [KServe](integrations/kserve.md)
* [kubernetes-sigs/lws](frameworks/lws.md)
* [meta-llama/llama-stack](integrations/llamastack.md)
* [substratusai/kubeai](integrations/kubeai.md)
* [vllm-project/aibrix](https://github.com/vllm-project/aibrix)
* [vllm-project/production-stack](integrations/production-stack.md)

## 使用 CPU 部署

!!! note
    此处使用 CPU 仅用于演示和测试目的，其性能无法与 GPU 相媲美。

首先，创建一个 Kubernetes PVC 和 Secret 用于下载和存储 Hugging Face 模型：

```bash
cat <<EOF |kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-models
spec:
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
type: Opaque
data:
  token: $(HF_TOKEN)
EOF
```

接下来，将 vLLM 服务器作为 Kubernetes Deployment 和 Service 启动：

```bash
cat <<EOF |kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: vllm
  template:
    metadata:
      labels:
        app.kubernetes.io/name: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["/bin/sh", "-c"]
        args: [
          "vllm serve meta-llama/Llama-3.2-1B-Instruct"
        ]
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        ports:
          - containerPort: 8000
        volumeMounts:
          - name: llama-storage
            mountPath: /root/.cache/huggingface
      volumes:
      - name: llama-storage
        persistentVolumeClaim:
          claimName: vllm-models
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector:
    app.kubernetes.io/name: vllm
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
EOF
```

我们可以通过查看日志验证 vLLM 服务器是否成功启动（下载模型可能需要几分钟）：

```console
kubectl logs -l app.kubernetes.io/name=vllm
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 使用 GPU 部署

**前提条件**：确保您有一个运行中的[支持 GPU 的 Kubernetes 集群](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)。

1. 为 vLLM 创建 PVC、Secret 和 Deployment

      PVC 用于存储模型缓存，它是可选的，您可以使用 hostPath 或其他存储选项。

      ```yaml
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ```

      Secret 是可选的，仅在访问受限模型时需要，如果您不使用受限模型，可以跳过此步骤。

      ```yaml
      apiVersion: v1
      kind: Secret
      metadata:
        name: hf-token-secret
        namespace: default
      type: Opaque
      stringData:
        token: "REPLACE_WITH_TOKEN"
      ```

      接下来创建 vLLM 的 Deployment 文件以运行模型服务器。以下示例部署 `Mistral-7B-Instruct-v0.3` 模型。

      以下是使用 NVIDIA GPU 和 AMD GPU 的两个示例。

      NVIDIA GPU：

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM 需要访问主机的共享内存以进行张量并行推理。
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            containers:
            - name: mistral-7b
              image: vllm/vllm-openai:latest
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HUGGING_FACE_HUB_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  nvidia.com/gpu: "1"
                requests:
                  cpu: "2"
                  memory: 6G
                  nvidia.com/gpu: "1"
              volumeMounts:
              - mountPath: /root/.cache/huggingface
                name: cache-volume
              - name: shm
                mountPath: /dev/shm
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 10
              readinessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 5
      ```

      AMD GPU：

      如果使用 AMD ROCm GPU（如 MI300X），可以参考下面的 `deployment.yaml`。

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            # PVC
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM 需要访问主机的共享内存以进行张量并行推理。
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "8Gi"
            hostNetwork: true
            hostIPC: true
            containers:
            - name: mistral-7b
              image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
              securityContext:
                seccompProfile:
                  type: Unconfined
                runAsGroup: 44
                capabilities:
                  add:
                  - SYS_PTRACE
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-v0.3 --port 8000 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
              ]
              env:
              - name: HUGGING_FACE_HUB_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  amd.com/gpu: "1"
                requests:
                  cpu: "6"
                  memory: 6G
                  amd.com/gpu: "1"
              volumeMounts:
              - name: cache-volume
                mountPath: /root/.cache/huggingface
              - name: shm
                mountPath: /dev/shm
      ```

      您可以从 <https://github.com/ROCm/k8s-device-plugin/tree/master/example/vllm-serve> 获取完整的示例、步骤和示例 yaml 文件。

2. 为 vLLM 创建 Kubernetes Service

      接下来，创建 Kubernetes Service 文件以暴露 `mistral-7b` 部署：

      ```yaml
      apiVersion: v1
      kind: Service
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        ports:
        - name: http-mistral-7b
          port: 80
          protocol: TCP
          targetPort: 8000
        # 标签选择器应与部署标签匹配，对于前缀缓存功能很有用
        selector:
          app: mistral-7b
        sessionAffinity: None
        type: ClusterIP
      ```

3. 部署和测试

      使用 `kubectl apply -f <filename>` 应用部署和服务配置：

      ```console
      kubectl apply -f deployment.yaml
      kubectl apply -f service.yaml
      ```

      要测试部署，运行以下 `curl` 命令：

      ```console
      curl http://mistral-7b.default.svc.cluster.local/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "mistralai/Mistral-7B-Instruct-v0.3",
              "prompt": "San Francisco is a",
              "max_tokens": 7,
              "temperature": 0
            }'
      ```

      如果服务正确部署，您应该会收到来自 vLLM 模型的响应。

## 结论

使用 Kubernetes 部署 vLLM 可以高效地扩展和管理利用 GPU 资源的机器学习模型。通过遵循上述步骤，您应该能够在 Kubernetes 集群中设置和测试 vLLM 部署。如果您遇到任何问题或有建议，请随时为文档做出贡献。