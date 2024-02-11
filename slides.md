---
marp: true
theme: default
paginate: true
author: Romain Clement & Pierre-Loïc Bayart
title: Running machine learning models in the browser
description: Learn how to run inference of machine learning and deep learning models locally in the browser
url: https://github.com/pieandai-grenoble/2024-02-23-ml-models-web
---

![bg](static/banner.png)

---

## Summary

1. 🧠 ML workflow recap
2. 🤝 ONNX format
3. 🌍 WebAssembly
4. 🧑‍💻 Examples
5. 🏁 Final notes

---

## What's all about?

<!-- Quick demo of image classification with a cat -->

---

## 🧠 ML workflow recap

1. Model training
2. Model inference

<!-- TODO: add illustration of supervised learning -->

---

## 🧠 ML workflow recap - Model training

- Offline
- Data collection
- Compute intensive
- Python, R, Julia, MATLAB, etc.

<!-- TODO: add logos -->

---

## 🧠 ML workflow recap - Model inference

- Online runtime
- Predictions from trained model
- Less compute intensive
- Python, C++, Go, Rust, etc.

<!-- Mention while it may be less compute intensive than training, it can become power hungry when used at scale with lots of users (cf. OpenAI) -->

<!-- TODO: add logos -->

---

## 🤝 ONNX format

[Open Neural Network Exchange][onnx]

- Generic ML model representation
- Common file format
- Training / inference loose coopling
- Interoperability
- Inference in any language
- Inference on multiple backends

<!-- Note: while the name implies NN, it can be used for any type of model not only NNs -->

<!-- TODO: add illustration of Netron graph -->

<!-- TODO: add illustration from model training to generic inference -->
![bg right]()

---

## 🤝 ONNX format

Export models from favourite framework:

- Scikit-Learn: [sklearn-onnx][sklearn-onnx]
- Tensorflow: [tensorflow-onnx][tensorflow-onnx]
- PyTorch: [torch.onnx][torch-onnx]

> ⚠️ Some models or layer types might not be supported by generic [operators][onnx-operators] yet!

---

## 🤝 ONNX format

Using [Netron][netron] to visualize an ONNX model

<!-- TODO: add Netron screenshot of simple model -->
![bg right]()

---

## 🤝 ONNX format

Available runtimes:

* C/C++
* Python
* ...
* **Web**!

---

## 🌍 WebAssembly

[_WASM_][webassembly]

- Portable compilation target
- Client and server applications
- Major browsers support (desktop, mobile)
- Fast, safe and open
- Privacy

<!-- Note: supported by all major browsers since 2017 -->

<!-- Privacy: no personal information leaking server-side (cf. OpenAI) -->

---

## 🌍 WebAssembly

Famous usage in Data Science ecosystem:

* [Pyodide][pyodide] (_Python in browser_)
* [JupyterLite][jupyterlite] (_JupyterLab in browser_)
* [PyScript][pyscript] (_Python in HTML_)
* **ONNX Runtime Web**!

---

## ⚙️ How does it work?



---

## 🧑‍💻 Examples

🏠 Housing Value Estimation

- [Web](samples/housing/index.html)
- [Notebook](samples/housing/training.html)

---

## 🧑‍💻 Examples

🍿 Sentiment Analysis

- [Web](samples/sentiment/index.html)
- [Notebook](samples/sentiment/training.html)

---

## 🧑‍💻 Examples

[🌉 Image Classification](samples/imaging/index.html)

- [Web](samples/imaging/index.html)
- [Notebook](samples/imaging/training.html)

---

## 🏁 Final notes

✨ Pros

- Inference at the edge
- No server required for inference
- Easier app integration
- Leverage WebGPU API (experimental)

☢️ Cons

- Lack of documentation
- Memory limitations
- Model size

---

## 📚 References

- [ONNX][onnx]
- [ONNX Runtime][onnx-runtime]
- [ONNX Runtime Web Samples][onnx-runtime-web-samples]
- [Netron][netron]
- [sklearn-onnx][sklearn-onnx]
- [tensorflow-onnx][tensorflow-onnx]
- [torch.onnx][torch-onnx]
- [WebAssembly][webassembly]

[onnx]: https://onnx.ai
[onnx-operators]: https://onnx.ai/onnx/operators/
[onnx-runtime]: https://onnxruntime.ai
[onnx-runtime-web-samples]: https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/importing_onnxruntime-web
[netron]: https://netron.app
[sklearn-onnx]: https://onnx.ai/sklearn-onnx/
[tensorflow-onnx]: https://github.com/onnx/tensorflow-onnx
[torch-onnx]: https://pytorch.org/docs/stable/onnx.html
[webassembly]: https://webassembly.org
[pyodide]: https://pyodide.org
[jupyterlite]: https://jupyterlite.readthedocs.io
[pyscript]: https://pyscript.net