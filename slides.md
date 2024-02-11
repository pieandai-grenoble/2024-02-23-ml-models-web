---
marp: true
theme: default
paginate: true
author: Romain Clement & Pierre-Loïc Bayart
title: Running machine learning models in the browser
description: Learn how to run inference of machine learning and deep learning models locally in the browser
url: https://github.com/pieandai-grenoble/2024-02-23-ml-models-web
---

![bg](banner.png)

---

## Summary

1. 🧠 ML workflow recap
2. 🤝 ONNX format
3. 🌍 WebAssembly
4. 🧑‍💻 Examples
5. 🏁 Final notes

---

## 🧠 ML workflow recap

1. Model training
2. Model inference

---

## 🧠 ML workflow recap - Model training

- Offline
- Data collection
- Compute intensive
- Python, R, Julia, MATLAB, etc.

---

## 🧠 ML workflow recap - Model inference

- Online runtime
- Predictions from trained model
- Less compute intensive
- Python, C++, Go, Rust, etc.

---

## 🤝 ONNX format

[Open Neural Network Exchange][onnx]

- Generic ML model representation
- Common file format
- Training / inference loose coopling
- Interoperability
- Inference in any language
- Inference on multiple backends

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

---

## 🌍 WebAssembly

Famous usage in Data Science ecosystem:

* [Pyodide][pyodide] (_Python in browser_)
* [JupyterLite][jupyterlite] (_JupyterLab in browser_)
* [PyScript][pyscript] (_Python in HTML_)
* **ONNX Runtime Web**!

---

## 🧑‍💻 Examples

---

## 🏁 Final notes

✨ Pros

- Inference at the edge
- No server required for inference
- Easier app integration
- Leverage WebGPU API

☢️ Cons

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