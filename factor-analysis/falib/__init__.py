"""falib: exploratory factor analysis over the model x trait matrix.

Pure functions, no file I/O (that lives in run_fa.py). Modules:
  config     FAConfig — every parameter with its default, JSON round-trip
  parcel     merge near-duplicate traits into unidimensional parcels
  adequacy   Bartlett sphericity, KMO
  retention  parallel analysis, Kaiser, Velicer MAP
  extract    principal-axis factoring
  rotate     varimax, promax (pattern, structure, factor correlations)
  scores     regression factor scores (orthogonal / oblique)
  pipeline   run_fa(matrix, config) -> FAResult
"""
