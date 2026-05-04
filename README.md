# generalizing-deepfake-audio-detection
Generalizing Deepfake Audio Detection (WIP)

paper coming in a few days.

<br/>

architecture overview from paper:

As shown in Figure 1, FSDAD (few-shot deepfake audio detection) has two training phases: embedder training and prototype memory
initialization, which are described in 2.2 and 2.3. After the prototype memory (2.4) is initialized with
examples from the training dataset, support sets are added through clustering and rejection in section
2.5. Finally, the classifier head is described in 2.6.

<br/>

intro from paper:

Modern deepfake audio detection models such as wav2vec2 feature extractors (Baevski et al.)
with classifier heads (Zhang et al.) inherently rely on the features of deepfake generation models
in their training datasets to classify audio. Müller et al. suggest that the failing generalization of
state-of-the-art models may occur because of this, with models over-tailoring to specific models,
datasets or audio environments (Müller et al.)

To address this, we introduce few-shot deepfake audio detection (FSDAD), an explainable prototype
learning-based framework for few-shot classification (detection) of audio deepfakes. The model
stores embeddings of deepfake artifacts in a sparse HNSW-based prototype memory (Malkov and
Yashunin), which are compared against embeddings of input audio to determine audio fakeness.
This approach has two main benefits: (1) accuracy generalizes to new deepfake models with only a
few examples added to the prototype memory, and (2) the usage of a sparse HNSW-based prototype
memory yields memory updates in O(NlogN), allowing the model’s memory to
efficiently scale.
