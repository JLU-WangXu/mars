# MARS-FIELD Related Work and References Draft v1

## Related work

`MARS-FIELD` sits at the intersection of structure-guided protein design, evolutionary prior integration, retrieval-based structural memory, and learned controller-decoder systems. The most relevant prior work can be divided into four categories.

### Structure prediction and structure-conditioned sequence modeling

Modern structure-aware protein modeling has been transformed by the success of large-scale structure prediction and inverse-folding systems. AlphaFold established that high-accuracy atomic structure prediction can be achieved at scale, thereby providing a structural substrate for downstream design workflows [1]. ProteinMPNN subsequently demonstrated the power of structure-conditioned sequence design, showing that neural inverse folding can generate high-quality sequences compatible with fixed backbones [2]. Related inverse-folding approaches, including large-scale learning from predicted structures, further expanded the scope of structure-conditioned sequence modeling [3]. These works strongly motivate the use of structure-conditioned information in design, but they do not by themselves define how heterogeneous engineering evidence should be unified when the design objective extends beyond structure compatibility alone.

### Retrieval-based structural comparison

Foldseek and related fast structural-search methods made it possible to retrieve local or global structural analogues at large scale [5]. For protein engineering, this retrieval capacity is highly valuable because it provides access to motif-level or neighborhood-level structural memory. However, retrieval alone does not define how structural neighbors should be transformed into a learned design controller. In `MARS-FIELD`, retrieval is therefore not treated as a stand-alone search module, but as a retrieval-memory branch whose outputs enter a shared residue-field representation.

### Sequence-based protein representation learning

Sequence-based deep representation learning has already shown that general protein embeddings can support rational engineering and transfer across functional contexts [6]. These approaches established that neural protein representations can capture useful evolutionary and biochemical regularities. However, many sequence-only approaches remain relatively weak at representing residue-level structural context, local pairwise couplings, or engineering-specific constraints such as oxidation hotspots and surface liabilities. `MARS-FIELD` differs by explicitly combining structural, evolutionary, ancestral, retrieval, and engineering context in a residue-level control framework.

### Ancestral sequence reconstruction and lineage-aware learning

Ancestral sequence reconstruction has long been recognized as a powerful source of stability and functional insight in protein evolution and engineering [7]. More recently, ancestry has also been used as a learning signal in modern representation-learning frameworks [8]. These developments are especially relevant to `MARS-FIELD`, because the current controller includes an explicit ancestral-lineage branch and a lineage-memory mechanism rather than treating ASR as a one-off heuristic recommendation. This allows ancestry to contribute as a structured prior within the controller rather than as an isolated preprocessing artifact.

## Positioning of MARS-FIELD

The present work differs from prior pipelines in one central way: it treats heterogeneous evidence as input to a shared residue-field controller rather than as a set of independent proposal or filtering tools. In that sense, `MARS-FIELD` is closer to an evidence-conditioned control system than to a conventional structure-plus-scoring pipeline.

At the same time, we do not yet claim a fully joint protein design foundation model. The current implementation still uses staged supervision and a hybrid final policy. The contribution of `MARS-FIELD` is therefore not that it completes the final end-state of joint protein design, but that it demonstrates a credible intermediate architecture in which:

- multi-modal evidence is unified in a shared residue field
- a learned controller performs calibrated candidate selection
- a decode-time neural branch is active in the benchmark-time main path
- the resulting system remains broadly benchmark-stable

This places `MARS-FIELD` as a bridge between traditional engineering stacks and future fully joint field-generator-decoder systems.

## Selected references

1. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583-589 (2021).

2. Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. *Science* **378**, 49-56 (2022).

3. Hsu, C. et al. Learning inverse folding from millions of predicted structures. In *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 8946-8970 (2022).

4. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123-1130 (2023).

5. van Kempen, M. et al. Fast and accurate protein structure search with Foldseek. *Nature Biotechnology* **42**, 243-246 (2024).

6. Alley, E. C. et al. Unified rational protein engineering with sequence-based deep representation learning. *Nature Methods* **16**, 1315-1322 (2019).

7. Chisholm, L. O. et al. Ancestral reconstruction and the evolution of protein energy landscapes. *Annual Review of Biophysics* **53**, 127-146 (2024).

8. Matthews, D. S. et al. Leveraging ancestral sequence reconstruction for protein representation learning. *Nature Machine Intelligence* **6**, 1542-1555 (2024).
