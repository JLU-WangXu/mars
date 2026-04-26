# MARS-FIELD Related Work and References v2

## Related work

### Structure prediction and structure-conditioned sequence design

Recent progress in protein machine learning has established that structure-aware representations can be learned and deployed at scale. AlphaFold demonstrated that high-accuracy atomic structure prediction can be achieved broadly enough to reshape downstream design workflows [1]. ProteinMPNN subsequently showed that fixed-backbone sequence design can be addressed effectively with deep structure-conditioned models [2]. Related inverse-folding systems trained on large collections of predicted structures further expanded the scope of structure-aware sequence generation [3]. These works are foundational for `MARS-FIELD` because they motivate both the use of structure-conditioned compatibility and the idea that residue-level geometric context should enter the controller natively rather than only through human-crafted filters.

At the same time, these systems are not themselves complete protein-engineering controllers. They primarily address backbone-conditioned sequence plausibility. In practical engineering settings, additional evidence streams are often required, including family-differential constraints, ancestor-derived priors, structure-local motif retrieval, and environment-specific engineering context. `MARS-FIELD` is positioned as a controller architecture that attempts to unify these additional streams rather than treating structure-conditioned sequence generation as the sole design primitive.

### Retrieval-based structural memory

Fast structural search methods, especially Foldseek, made it practical to retrieve local and global structural analogues at scale [5]. Retrieval is highly relevant to protein engineering because it offers motif-level memory that can anchor design decisions in empirically observed structural neighborhoods. However, retrieval-based support is often used as an auxiliary annotation or post hoc sanity check rather than as a learned part of the controller. `MARS-FIELD` differs by treating retrieval as a structured evidence stream that enters both the engineering field and the neural controller through a dedicated retrieval-memory branch.

### Sequence-based protein representation learning and rational engineering

Deep sequence representations have already shown substantial value for rational protein engineering [6]. These models demonstrate that large-scale sequence statistics encode useful biochemical and evolutionary regularities, and that sequence-only embeddings can support meaningful transfer. However, sequence representations alone do not fully resolve how to combine geometric structure, retrieval memory, and explicit engineering constraints in a unified controller. `MARS-FIELD` therefore sits closer to a multi-evidence control system than to a pure sequence language-model application.

### Evolutionary priors and ancestral sequence reconstruction

Ancestral sequence reconstruction has long played an important role in protein evolution studies and protein engineering [7]. More recently, ancestry has also become relevant to modern representation learning and machine learning for proteins [8]. This line of work is especially important for `MARS-FIELD`, because the method does not treat ASR as a one-off recommendation list. Instead, ancestry is represented as a dedicated lineage-aware branch and contributes both explicit residue priors and learned latent structure within the controller. This allows the method to capture not only what residues are plausible, but also how lineage-derived uncertainty should shape the field.

### Positioning of MARS-FIELD

The central distinction of `MARS-FIELD` is that it treats heterogeneous evidence as input to a shared residue-field controller rather than as a set of separate candidate generators, filters, or post hoc heuristics. In this sense, the method is closer to an evidence-conditioned controller-decoder architecture than to a conventional structure-plus-ranking pipeline.

At the same time, the present implementation should not be overstated. It is not yet a fully joint proposal-generator / field / decoder optimization framework. Instead, it is best described as an intermediate but meaningful architecture in which:

- structural, evolutionary, ancestral, retrieval, and environmental evidence are unified in a residue field
- a neural controller performs candidate-level calibrated ranking
- a decode-time neural field generator is active in the main benchmark path
- the resulting system remains benchmark-stable at the panel level

In this sense, `MARS-FIELD` should be viewed as a bridge between heuristic engineering stacks and future fully joint field-generator-decoder systems.

## Selected references

1. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583-589 (2021).

2. Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. *Science* **378**, 49-56 (2022).

3. Hsu, C. et al. Learning inverse folding from millions of predicted structures. In *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 8946-8970 (2022).

4. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* **379**, 1123-1130 (2023).

5. van Kempen, M. et al. Fast and accurate protein structure search with Foldseek. *Nature Biotechnology* **42**, 243-246 (2024).

6. Alley, E. C. et al. Unified rational protein engineering with sequence-based deep representation learning. *Nature Methods* **16**, 1315-1322 (2019).

7. Chisholm, L. O. et al. Ancestral reconstruction and the evolution of protein energy landscapes. *Annual Review of Biophysics* **53**, 127-146 (2024).

8. Matthews, D. S. et al. Leveraging ancestral sequence reconstruction for protein representation learning. *Nature Machine Intelligence* **6**, 1542-1555 (2024).
