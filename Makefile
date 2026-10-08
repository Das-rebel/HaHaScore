# HaHaScore Makefile
# Improvement workflow - reproducible from clean checkout.

.PHONY: help install ci data train eval clean all

help:
	@echo "HaHaScore Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install  Install Python deps"
	@echo "  ci       Run continuous-integration (laugho_ci.py)"
	@echo "  data     Build laugh dataset (laugho_data.py --decode)"
	@echo "  train    Train v2 LaughO model (v2_laugho_train.py)"
	@echo "  eval     Run 5×3 repeated CV (laugho_cv.py)"
	@echo "  paper    Compile LaTeX paper"
	@echo "  clean    Remove build artifacts"

install:
	pip install -r requirements.txt

ci:
	python3 laugho_ci.py

data:
	python3 laugho_data.py --shards data/eval/00.parquet --decode --max-rows 5000

train:
	python3 v2_laugho_train.py --features experiments/v2_laugho/laughs.npz --epochs 10

eval:
	python3 laugho_cv.py

paper:
	cd arxiv_submission && tar -czf hahascore_arxiv_bundle.tar.gz \
	    hahascore.tex \
	    references.bib \
	    figure_5x3_falsification.pdf \
	    figure_speaker_disjoint.pdf \
	    README.md \
	    COVER_LETTER.md
	@echo "Bundle: arxiv_submission/hahascore_arxiv_bundle.tar.gz"
	cd arxiv_submission && tar -czf v1_paper_arxiv_bundle.tar.gz \
	    v1_paper.tex \
	    figure_v1_5x3.png \
	    figure_v1_5x3.pdf \
	    references.bib \
	    v1_README.md
	@echo "Bundle: arxiv_submission/v1_paper_arxiv_bundle.tar.gz""
	@echo "Bundle: arxiv_submission/v1_paper_arxiv_bundle.tar.gz"

all: ci eval paper
	@echo "Full validation complete. Bundle ready for upload."

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf experiments/*/checkpoints/*.pt
	rm -rf arxiv_submission/hahascore_arxiv_bundle.tar.gz