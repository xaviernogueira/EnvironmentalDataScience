"""
Uses HyRiver's py3dep to get Digital Elevation Model (DEM) data for an AOI, from which the following 
elevation based features can be built and sampled for ML. 
    * mean elevation - mean_z (by resolution size)
    * std elevation - std_z (outputs to mean_z resolution size, but calculated across a kernal resolution)
    * relative elevation - rel_z = (elevation of a cell - mean kernal elevation) / kernal elevation std (aka a z-score)
    * surrounding topographic roughness - roughness_z = std_z / mean_z
Many more can and should be added. Models are as good as their features.
"""