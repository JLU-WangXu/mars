# MARS-FIELD Related Work and References v3

## Related work

### Structure prediction and inverse folding

Modern protein design pipelines were transformed by large-scale advances in structure prediction and structure-conditioned sequence modeling. AlphaFold established that accurate atomic structure prediction can be achieved broadly enough to reshape downstream design workflows [1]. ProteinMPNN then showed that fixed-backbone sequence design can be solved effectively with deep structure-conditioned models [2], while large-scale inverse-folding systems trained on predicted structures further expanded the design setting [3]. These studies motivate the use of geometry-conditioned compatibility as a native modeling object, but they do not by themselves resolve how multiple non-geometric evidence streams should be unified in engineering contexts.

### Retrieval-based structural memory

Foldseek and related fast structural-search systems made motif-level structural retrieval practical at large scale [5]. Such retrieval is highly valuable for protein engineering because it provides access to empirically grounded structural neighborhoods. However, retrieval is often used as an external annotation or post hoc check rather than as a learned part of a design controller. `MARS-FIELD` differs by treating retrieval as a structured evidence stream that enters both the residue field and the neural controller through a dedicated memory-linked branch.

### Sequence-based representation learning and rational engineering

Sequence-based representation learning has already shown that large protein embeddings can support rational engineering and transfer across tasks [6]. These methods demonstrate that sequence statistics encode useful biochemical and evolutionary constraints. However, sequence-only representations do not, on their own, provide a natural mechanism for combining local structural context, retrieval memory, ancestry, and explicit engineering penalties in a unified controller. `MARS-FIELD` therefore sits closer to an evidence-conditioned controller-decoder framework than to a pure sequence language-model application.

### Ancestral sequence reconstruction and lineage-aware learning

Ancestral sequence reconstruction has long been important in protein evolution and engineering [7]. More recently, ancestry has also been used as a learning signal in representation-learning systems [8]. This literature is particularly relevant because `MARS-FIELD` treats ancestry not as a one-off recommendation list but as a dedicated lineage-aware branch that contributes both explicit residue priors and latent controller structure.

### Positioning of MARS-FIELD

The central distinction of `MARS-FIELD` is that it treats heterogeneous evidence as input to a shared residue-field controller rather than as a set of independent proposal engines or post hoc heuristics. The current implementation should not be overstated as a fully joint proposal-generator / field / decoder optimization framework. Instead, it is best viewed as an intermediate but meaningful architecture in which:

- structural, evolutionary, ancestral, retrieval, and environmental evidence are unified in a residue field
- a neural controller performs calibrated candidate-level selection
- a decode-time neural field generator is active in the benchmark-time main path
- the resulting system remains benchmark-stable at the panel level

In this sense, `MARS-FIELD` functions as a bridge between heuristic engineering stacks and future fully joint field-generator-decoder systems.

## Selected references

1. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583-589 (2021).

2. Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. *Science* **378**, 49-56 (2022).

3. Hsu, C. et al. Learning inverse folding from millions of predicted structures. In *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 8946-8970 (2022).

4. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123-1130 (2023).

5. van Kempen, M. et al. Fast and accurate protein structure search with Foldseek. *Nature Biotechnology* **42**, 243-246 (2024).

6. Alley, E. C. et al. Unified rational protein engineering with sequence-based deep representation learning. *Nature Methods* **16**, 1315-1322 (2019).

7. Chisholm, L. O. et al. Ancestral reconstruction and the evolution of protein energy landscapes. *Annual Review of Biophysics* **53**, 127-146 (2024).

8. Matthews, D. S. et al. Leveraging ancestral sequence reconstruction for protein representation learning. *Nature Machine Intelligence* **6**, 1542-1555 (2024).
