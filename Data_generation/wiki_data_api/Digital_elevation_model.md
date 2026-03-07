# Digital elevation model

A digital elevation model (DEM) or digital surface model (DSM) is a 3D computer graphics representation of elevation data to represent terrain or overlaying objects, commonly of a planet, moon, or asteroid. A "global DEM" refers to a  discrete global grid. DEMs are used often in geographic information systems (GIS), and are the most common basis for digitally produced relief maps.

A digital terrain model (DTM) represents specifically the ground surface while DEM and DSM may represent tree top canopy or building roofs.

While a DSM may be useful for landscape modeling, city modeling and visualization applications, a DTM is often required for flood or drainage modeling, land-use studies, geological applications, and other applications, and in planetary science.

There is no universal usage of the terms digital elevation model (DEM), digital terrain model (DTM) and digital surface model (DSM) in scientific literature. In most cases the term digital surface model represents the earth's surface and includes all objects on it. In contrast to a DSM, the digital terrain model (DTM) represents the bare ground surface without any objects like plants and buildings (see the figure on the right).

DEM is often used as a generic term for DSMs and DTMs, only representing height information without any further definition about the surface.

Other definitions equalise the terms DEM and DTM, equalise the terms DEM and DSM,

define the DEM as a subset of the DTM, which also represents other morphological elements, or define a DEM as a rectangular grid and a DTM as a three-dimensional model (TIN).

Most of the data providers (USGS, ERSDAC, CGIAR, Spot Image) use the term DEM as a generic term for DSMs and DTMs. Some datasets such as SRTM or the ASTER GDEM are originally DSMs, although in forested areas, SRTM reaches into the tree canopy giving readings somewhere between a DSM and a DTM). DTMs are created from high resolution DSM datasets using complex algorithms to filter out buildings and other objects, a process known as "bare-earth extraction".

In the following, the term DEM is used as a generic term for DSMs and DTMs.

A DEM can be represented as a raster (a grid of squares, also known as a heightmap when representing elevation) or as a vector-based triangular irregular network (TIN). The TIN DEM dataset is also referred to as a primary (measured) DEM, whereas the Raster DEM is referred to as a secondary (computed) DEM.  The DEM could be acquired through techniques such as photogrammetry, lidar, IfSAR or InSAR, land surveying, etc. (Li et al. 2005).

DEMs are commonly built using data collected using remote sensing techniques, but they may also be built from land surveying.

The digital elevation model itself consists of a matrix of numbers, but the data from a DEM is often rendered in visual form to make it understandable to humans.  This visualization may be in the form of a contoured topographic map, or could use shading and false color  assignment (or "pseudo-color") to render elevations as colors (for example, using green for the lowest elevations, shading to red, with white for the highest elevation.).

Visualizations are sometimes also done as oblique views, reconstructing a synthetic visual image of the terrain as it would appear looking down at an angle. In these oblique visualizations, elevations are sometimes scaled using "vertical exaggeration" in order to make subtle elevation differences more noticeable. Some scientists,

however, object to vertical exaggeration as misleading the viewer about the true landscape.

Mappers may prepare digital elevation models in a number of ways, but they frequently use remote sensing rather than direct survey data.

Older methods of generating DEMs often involve interpolating digital contour maps that may have been produced by direct survey of the land surface. This method is still used in mountain areas, where interferometry is not always satisfactory. Note that contour line data or any other sampled elevation datasets (by GPS or ground survey) are not DEMs, but may be considered digital terrain models. A DEM implies that elevation is available continuously at each location in the study area.

One powerful technique for generating digital elevation models is interferometric synthetic aperture radar where two passes of a radar satellite (such as RADARSAT-1 or TerraSAR-X or Cosmo SkyMed), or a single pass if the satellite is equipped with two antennas (like the SRTM instrumentation), collect sufficient data to generate a digital elevation map tens of kilometers on a side with a resolution of around ten meters. Other kinds of stereoscopic pairs can be employed using the digital image correlation method, where two optical images are acquired with different angles taken from the same pass of an airplane or an Earth Observation Satellite (such as the HRS instrument of SPOT5 or the VNIR band of ASTER).

The SPOT 1 satellite (1986) provided the first usable elevation data for a sizeable portion of the planet's landmass, using two-pass stereoscopic correlation. Later, further data were provided by the European Remote-Sensing Satellite (ERS, 1991) using the same method, the Shuttle Radar Topography Mission (SRTM, 2000) using single-pass SAR and the Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER, 2000) instrumentation on the Terra satellite using double-pass stereo pairs.

The HRS instrument on SPOT 5 has acquired over 100 million square kilometers of stereo pairs.

A tool of increasing value in planetary science has been use of orbital altimetry used to make digital elevation map of planets.  A primary tool for this is laser altimetry but radar altimetry is also used. Planetary digital elevation maps made using laser altimetry include the Mars Orbiter Laser Altimeter (MOLA) mapping of Mars, the Lunar Orbital Laser Altimeter (LOLA) and Lunar Altimeter (LALT) mapping of the Moon, and the Mercury Laser Altimeter (MLA) mapping of Mercury. In planetary mapping, each planetary body has a unique reference surface. New Horizons' Long Range Reconnaissance Imager used stereo photogrammetry to produce partial surface elevation maps of Pluto and 486958 Arrokoth.

Methods for obtaining elevation data used to create DEMs

Structure from motion / Multi-view stereo applied to aerial photography

The quality of a DEM is a measure of how accurate elevation is at each pixel (absolute accuracy) and how accurately is the morphology presented (relative accuracy). Quality assessment of DEM can be performed by comparison of DEMs from different sources. Several factors play an important role for quality of DEM-derived products:

sampling density (elevation data collection method);

Reference 3D products include quality masks that give information on the coastline, lake, snow, clouds, correlation etc.

Modeling water flow for hydrology or mass movement (for example avalanches and landslides)

Modeling soils wetness with Cartographic Depth to Water Indexes (DTW-index)

Creation of physical models (including raised relief maps and 3D printed terrain models)

Rectification of aerial photography or satellite imagery

Reduction (terrain correction) of gravity measurements (gravimetry, physical geodesy)

Terrain analysis in geomorphology and physical geography

Auto safety / advanced driver-assistance systems (ADAS)

Released at the beginning of 2022, FABDEM offers a bare earth simulation of the Earth's surface at 1 arc-second resolution (about 30 meters). Adapted from GLO-30, the data removes all forests and buildings. The data is free to download non-commercially and through the developer's website at a cost commercially.

An alternative free global DEM is called GTOPO30 (30 arcsecond resolution, c. 1 km  along the equator) is available, but its quality is variable and in some areas it is very poor. A much higher quality DEM from the Advanced Spaceborne Thermal Emission and Reflection Radiometer (ASTER) instrument of the Terra satellite is also freely available for 99% of the globe, and represents elevation at 30 meter resolution. A similarly high resolution was previously only available for the United States territory under the Shuttle Radar Topography Mission (SRTM) data, while most of the rest of the planet was only covered in a 3 arc-second resolution (around 90 meters  along the equator). SRTM does not cover the polar regions and has mountain and desert no data (void) areas. SRTM data, being derived from radar, represents the elevation of the first-reflected surface—quite often tree tops.  So, the data are not necessarily representative of the ground surface, but the top of whatever is first encountered by the radar.

Submarine elevation (known as bathymetry) data is generated using ship-mounted depth soundings. When land topography and bathymetry is combined, a truly global relief model  is obtained.   The SRTM30Plus dataset (used in NASA World Wind) attempts to combine GTOPO30, SRTM and bathymetric data to produce a truly global elevation model.  The Earth2014 global topography and relief model provides layered topography grids at 1 arc-minute resolution. Other than SRTM30plus, Earth2014 provides information on ice-sheet heights and bedrock (that is, topography below the ice) over Antarctica and Greenland. Another global model is Global Multi-resolution Terrain Elevation Data 2010 (GMTED2010) with 7.5 arc second resolution. It is based on SRTM data and combines other data outside SRTM coverage. A novel global DEM of postings lower than 12 m and a height accuracy of less than 2 m is expected from the TanDEM-X satellite mission which started in July 2010.

The most common grid (raster) spacing is between 50 and 500 meters. In gravimetry e.g., the primary grid may be 50 m, but is switched to 100 or 500 meters in distances of about 5 or 10 kilometers.

Since 2002, the HRS instrument on SPOT 5 has acquired over 100 million square kilometers of stereo pairs used to produce a DTED2  format DEM (with a 30-meter posting) DEM format DTED2 over 50 million km2. The radar satellite RADARSAT-2 has been used by MacDonald, Dettwiler and Associates Ltd. to provide DEMs for commercial and military customers.

In 2014, acquisitions from radar satellites TerraSAR-X and TanDEM-X will be available in the form of a uniform global coverage with a resolution of 12 meters.

ALOS provides since 2016 a global 1-arc second DSM free of charge, and a commercial 5 meter DSM/DTM.

Many national mapping agencies produce their own DEMs, often of a higher resolution and quality, but frequently these have to be purchased, and the cost is usually prohibitive to all except public authorities and large corporations. DEMs are often a product of national lidar dataset programs.

Free DEMs are also available for Mars: the MEGDR, or Mission Experiment Gridded Data Record, from the Mars Global Surveyor's Mars Orbiter Laser Altimeter (MOLA) instrument; and NASA's Mars Digital Terrain Model (DTM).

OpenTopography is a web based community resource for access to high-resolution, Earth science-oriented, topography data (lidar and DEM data), and processing tools running on commodity and high performance compute system along with educational resources. OpenTopography is based at the San Diego Supercomputer Center at the University of California San Diego and is operated in collaboration with colleagues in the School of Earth and Space Exploration at Arizona State University and UNAVCO. Core operational support for OpenTopography comes from the National Science Foundation, Division of Earth Sciences.

The OpenDemSearcher is a Mapclient with a visualization of regions with free available middle and high resolution DEMs.