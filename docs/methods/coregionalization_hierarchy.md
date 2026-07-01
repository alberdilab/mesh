# Hierarchical coregionalization: genomes, gene families and traits

:::{admonition} Design document — not yet implemented
:class: important

This page specifies a **later milestone**. It extends the existing linear model
of coregionalization ({func}`mesh.coregionalized_negbinomial`) from a *flat*
feature×field loadings matrix to a **structured** one, where the loadings are
built from the biological organisation of the features: which **genome** carries
a gene, which **gene family** it belongs to, and which **traits** (KEGG modules,
MetaCyc pathways, biosynthetic gene clusters) it contributes to. It is written
before the code so the modelling choices can be reviewed.
:::

## Why a grouping *level* is not enough

The current coregionalization model gives every feature its own free row in a
`(n_features × n_fields)` loadings matrix (see {doc}`model`). Two genes are free
to load on completely different fields. Biologically that is wrong in a specific,
consequential way: **genes physically co-located in one genome rise and fall
together** because they are copies carried by the same cell. Their per-field
loadings are therefore near-identical, and any "gene–gene coregionalization" read
off two genes from the *same* genome is trivially ≈ 1 — it reports cell biology,
not ecology.

There are two ways one might "add the genome", and only one of them helps:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Interpretation
  - Effect
* - Genome as a **grouping level** (a hierarchical offset on the intercept,
    grouped by genome)
  - Absorbs genome-level *mean* abundance differences. But in MESH the spatial
    structure lives in the **fields and loadings**, not the intercept — so this
    captures *"genome A is more abundant than B"* and **nothing** about *"A and B
    occupy the same patches"*. Wrong lever.
* - Genome as an **entity with its own spatial field** (this design)
  - The genome carries its **own latent field** — where that organism lives — and
    every gene in it inherits that field through a near-deterministic positive
    **weight** (copy number ≈ 1 per gene). This is the piece a flat LMC or a
    hierarchical intercept cannot express.
```

The same argument applies one tier up: a **trait** (a module/pathway/BGC) is not
a genome and not a single gene family — it is a *set of different genes that
together perform one function*, and its spatial signal is worth a field of its
own.

## The membership lattice

The three functional groupings do **not** form a clean tree. A single gene is
simultaneously a member of one genome, one gene family, and *several* traits:

```{mermaid}
flowchart TD
    G["Genome<br/>(organism — where it lives)"] -->|carries| Gene
    F["Gene family<br/>(KO / Pfam — homologs across genomes)"] -->|groups| Gene
    T["Trait<br/>(KEGG module / MetaCyc pathway / BGC)"] -->|composed of many| Gene
    Gene["Gene instance<br/>(the leaf: one genome, one family, many traits)"]
```

```{list-table}
:header-rows: 1
:widths: 20 20 60

* - Level
  - Groups
  - Meaning
* - **Genome**
  - horizontally-unrelated genes of one organism
  - *Where does this organism live?* One field per genome (or per genome factor).
* - **Gene family**
  - homologs of the **same** gene across genomes (KO, Pfam)
  - *Does a function segregate beyond the organisms that carry it?* — residual,
    cross-genome.
* - **Trait**
  - **different** gene families that co-function (module, pathway, BGC)
  - *Is a multi-gene capacity spatially organised, and is it realised within one
    genome or assembled across the community?*
```

Because membership is **many-to-many** (a gene family belongs to many traits; a
trait spans many families; a family spans many genomes), the model cannot be
nested random effects. It is a **structured linear model of coregionalization**:
the loadings matrix is *composed* from membership incidence matrices rather than
left free.

## The structured latent

For gene instance $i$, carried by genome $g(i)$, in family $\varphi(i)$, and
contributing to the set of traits $\mathcal{T}(i)$, the latent spatial
contribution (added to the intercept and depth offset by the likelihood, exactly
as today) is:

$$
\text{latent}_i(\mathbf{x}) \;=\;
\underbrace{\eta_{g(i)}\, f^{\text{gen}}_{g(i)}(\mathbf{x})}_{\text{organism location}}
\;+\;
\underbrace{a_i\, f^{\text{fam}}_{\varphi(i)}(\mathbf{x})}_{\substack{\text{function beyond}\\\text{taxonomy (residual)}}}
\;+\;
\underbrace{\sum_{\tau \in \mathcal{T}(i)} c_{g(i),\tau}\, b_{i\tau}\, f^{\text{trait}}_{\tau}(\mathbf{x})}_{\text{multi-gene capacity}}
\;+\;\varepsilon_i .
$$

- Each $f^{\bullet}$ is a **unit-variance Matérn field** with its **own range**
  (patch size). Ranges are ordered/extent-bounded per level exactly as in the
  current LMC, to pin field identity and avoid runaway basins.
- $\eta_{g(i)}$ is a **single positive amplitude per genome**, inherited
  *identically* by every gene the genome carries. This is the entity: an organism
  is a bundle of genes that rise and fall together in space. Because the field is
  in the **log** link, all genes of a genome fluctuate identically regardless of
  copy number or length — those are **mean** effects (a constant multiplier ⇒ an
  additive shift in log space) absorbed by the per-feature intercept and the
  length/depth **offset**, *not* the spatial amplitude. So there is deliberately
  **no free per-gene weight**: a gene departing from $\eta_{g(i)} f^{\text{gen}}$
  is exactly the residual the family/trait terms below capture.
- $a_i$, $b_{i\tau}$ are **signed** loadings (read magnitudes for assignment,
  sign for co- vs anti-segregation), as in the flat model.

:::{admonition} DNA now, expression later
:class: note

The shared per-genome amplitude is correct for **DNA abundance**, where every
gene of an organism tracks it in lockstep. If MESH later ingests
**metatranscriptomics**, expression genuinely varies gene-by-gene, and this term
should become a per-gene amplitude (a hierarchical multiplier centred on
$\eta_{g}$). That is a deliberate future change, not an omission.
:::
- $c_{g,\tau} \in [0,1]$ is the **genome-inferred completeness** of trait $\tau$
  in genome $g$ — a **known** weight from upstream annotation (the "genome-
  inferred" part), gating a gene's trait contribution by how completely its
  genome actually encodes the trait. It is *not* estimated.

### Identifiability by additive residual shrinkage

The three terms are strongly confounded unless ordered by how much variance they
are *allowed* to claim. MESH resolves this the way the biology suggests:

1. the **genome** field claims the bulk (genes co-vary because they are one
   cell);
2. the **gene-family** field claims only what the genome fields leave — this
   residual *is* "function beyond taxonomy";
3. the **trait** field claims only what genome + family leave — the emergent,
   cross-genome functional signal.

Concretely this is hierarchical shrinkage: tighter priors on the higher
functional levels (`loadings_scale` decreasing family→trait), so a trait field
only survives if it explains structure the carriers cannot. Using a single
per-genome amplitude $\eta_g$ (rather than a free per-gene weight) keeps the
genome level identifiable and rigid: it claims one clean share of variance and
leaves an interpretable residual, instead of $n_{\text{features}}$ weakly-
identified weights that would drift the model back toward the flat LMC.

## What spatial data uniquely resolves

Annotation tells you a genome *encodes* a pathway. Only spatial data can tell you
**how the pathway is realised**, and this is the headline the trait level unlocks:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Realization
  - Spatial signature
* - **Within-genome**
  - The module's genes co-localize *because they are the same cell*: the genome
    field explains their co-occurrence; the trait field adds ~nothing.
* - **Distributed / community**
  - The pathway's steps sit in **different** genomes yet still co-localize: a
    non-trivial trait field over-and-above the genome fields — metabolic handoff,
    division of labour, assembly in a shared patch.
* - **Scattered**
  - Genes carry the annotation but show no spatial coherence: the function is not
    spatially organised.
```

## Reportable statistics

Each functional level reports the same three interpretable quantities, in MESH's
existing idiom (patch size + credible interval, a co-segregation readout, a
variance partition):

```{list-table}
:header-rows: 1
:widths: 26 74

* - Statistic
  - Per level
* - **Patch size**
  - the Matérn range of each field, with a credible interval — *"the
    nitrogen-fixation module segregates at ~X µm"*.
* - **Coregionalization matrix**
  - genome×genome / family / trait similarity from the loadings (normalized
    $WW^\top$ or cosine), with posterior uncertainty — *which organisms /
    functions co-occupy patches*.
* - **Realization / variance partition**
  - per feature, family and trait: variance split across genome / family / trait
    / residual — the within-genome-vs-distributed decomposition above. This is
    the clean form of the determinism index already surfaced from single fits.
```

## Cost

Exact GP cost is $O(n^3)$ in the **number of microsamples** $n$ — the fields are
drawn over the shared coordinates, so it is **independent of the number of genes,
families or traits**. Adding a functional tier adds *fields* (more GP draws over
the same coords), so the compute budget is set by the **total number of fields**,
not by the size of the annotation. The real constraint is **loadings
identifiability**, which is exactly why the additive residual shrinkage above
matters. No SPDE/sparse/variational machinery is introduced.

## Schema impact

Counts stay long-format (see {doc}`schema`), but membership can no longer live in
columns of the counts table, because trait membership is many-to-many. The
contract grows a small set of **annotation tables**, validated alongside the
counts table and keyed on `feature_id`:

```{list-table}
:header-rows: 1
:widths: 26 20 54

* - Table
  - Cardinality
  - Columns
* - `feature → genome`
  - 1 : 1
  - `feature_id`, `genome_id`
* - `feature → family`
  - n : 1
  - `feature_id`, `family_id`
* - `feature ↔ trait`
  - n : m
  - `feature_id`, `trait_id`
* - `genome × trait completeness`
  - —
  - `genome_id`, `trait_id`, `completeness` ∈ [0, 1]
```

Validation extends in MESH's loud-failure style: every `feature_id` in an
annotation table must exist in the counts catalog; `completeness` must be in
[0, 1]; a trait referenced in the incidence table must have completeness for each
genome that carries a member gene. All of this is **optional** — the flat
{func}`mesh.coregionalized_negbinomial` remains valid input with no annotation
tables.

## Pinned decisions and open choices

- **Each level's fields get their own Matérn ranges** (default). This is the more
  interpretable choice because per-level patch sizes *are* MESH's headline output.
  The cheaper alternative — genomes/traits **loading on a shared pool of community
  fields** (fewer distinct scales) — stays available as a fallback for
  compute-bound problems and is a runtime flag, not a separate model.
- **Completeness is a known input, not estimated.** If completeness is unavailable
  upstream, it defaults to presence/absence (0/1) from the incidence table.
- **The genome contributes one shared amplitude $\eta_g$**, not a free per-gene
  weight; copy number and length stay as mean offsets (see the latent above). The
  per-gene metatranscriptomics variant is a deliberate future change.
