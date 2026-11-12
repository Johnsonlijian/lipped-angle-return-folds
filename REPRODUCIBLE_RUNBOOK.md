# Reproduction

Use CPython 3.11, NumPy 2.4.3 and SciPy 1.17.1; install requirements.txt into an isolated environment. The vendored pure-Python PyCUFSM sources take precedence through src/r19env.py. No compiled solver extension is required. Set BLAS thread counts to one for comparable timing. No random data are generated.

1. Check the section-property repair with `python src/test_cutwp_fix.py`.
2. Recompute printed numerical values with `python src/qa_numbers.py` (uses saved derived tables, not new simulations).
3. Regenerate the GMNIA figure with `python src/make_final_gmnia.py`; other new plots use `python src/make_figures_v3.py`. Inherited tables and figure PDFs are preserved as separate baseline evidence; the full original native-source reconstruction is not distributed as an experimental dataset.
4. New elastic calculations follow identify_v3.py, regime_map.py + relabel_v3.py regime, at_crossover.py, corner_radius.py, channel_crossover.py/channel_crossover_c25.py + relabel_v3.py channel, member_dsm_v2.py + relabel_v3.py dsm + member_dsm_v3.py. Their parameters and deterministic outputs are in data/.
5. Each gmnia/<case>/ contains the exact S4R input, model metadata and extracted result JSON. To rerun a case, copy its .inp to a working directory and use your licensed Abaqus command with job=<input stem> interactive. The portable gmnia.py generator accepts ABAQUS_COMMAND for its executable and creates a local gmnia_runs/ directory. Original vendor solver and own derived data are separate from commercial executable files.
6. Resolved peaks: 43/48; baseline peak statistics: 33 cases. Five maximum-attained-load observations are lower bounds and cannot be pooled as resolved ultimate strengths. Read the post_peak flag before comparing ratios or mode shares. P_lip measures spatial displacement, not a formal nonlinear local/distortional class.

Expensive Abaqus jobs are not rerun by the lightweight QA. Script execution reproduces the saved numerical summaries but does not independently validate physical truth or experimental applicability. Author approval and repository deposit remain required before calling this a public archive.
