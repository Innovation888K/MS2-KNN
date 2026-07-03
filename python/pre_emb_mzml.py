from config import PipelineConfig
from spec2vec_emb import build_spec2vec_embeddings_from_mzml
from specEmb_emb import build_specEmb_embeddings_from_mzml
from binned_emb import build_binned_embeddings_from_mzml
from dreams_emb import build_dreams_embeddings_from_mzml
from msbert_emb import build_msbert_embeddings_from_mzml
from ms2deepscore_emb import build_ms2deepscore_embeddings_from_mzml
config = PipelineConfig()
mzml_path=config.query_mzml_path
#
# spec2vec_embs = build_spec2vec_embeddings_from_mzml(
#     mzml_path,
#     "out/example.spec2vec.npz",
#     config,
# )
#
# specemb_embs = build_specEmb_embeddings_from_mzml(
#     mzml_path,
#     "out/example.specEmb.npz",
#     config,
# )
#
binned_embs = build_binned_embeddings_from_mzml(
    mzml_path,
    "out/example.binned.npz",
    config,
    mode="binned",
)
#
# nl_binned_embs = build_binned_embeddings_from_mzml(
#     mzml_path,
#     "out/example.neutral_loss_binned.npz",
#     config,
#     mode="neutral_loss",
# )
#
# dreams_embs = build_dreams_embeddings_from_mzml(
#     config,
# )
#
# msbert_embs = build_msbert_embeddings_from_mzml(
#     mzml_path,
#     "out/example.msbert.npz",
#     config,
# )

# build_ms2deepscore_embeddings_from_mzml(
#     mzml_path=mzml_path,
#     output_path="out/example.ms2deepscore.npz",
#     config=config,
#     ms_level=2,
#     max_spectra=None,
# )
