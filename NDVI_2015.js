// Load the region of interest (ROI) for Luxembourg from the asset
var roi = ee.FeatureCollection('projects/ee-prabina001/assets/LUX_adm4');

// Set the map center to the ROI
Map.centerObject(roi, 6);
Map.addLayer(roi, {color: 'FF0000'}, 'ROI');

// Load Landsat 8 image collection for the year 2013
var Landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
  .filterDate('2015-01-01', '2015-12-31')
  .filterBounds(roi)
  .median();

// Select bands for NDVI calculation
var NIR = Landsat.select("B5"); // Use 'B5' for NIR
var RED = Landsat.select('B4');  // Use 'B4' for RED

// Calculate NDVI
var ndvi = NIR.subtract(RED).divide(NIR.add(RED)).rename('NDVI');
var NDVI = ndvi.clip(roi);

// Add NDVI layer to the map
var ndviparam = {min: -1, max: 1, palette: ['blue', 'yellow', 'green']};
Map.addLayer(NDVI, ndviparam, 'NDVI');

// Reduce the NDVI image to get the mean value
var meanNDVI = NDVI.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  scale: 30, // Use a scale suitable for Landsat data
  maxPixels: 1e13,
  bestEffort: true // Use bestEffort to aggregate at whatever scale results in 'maxPixels'
});

// Print the mean NDVI value to the console
print('Mean NDVI:', meanNDVI.get('NDVI'));

// Export the NDVI image to Google Drive with LUREF_Transverse_Mercator projection
Export.image.toDrive({
  image: NDVI,
  description: 'NDVI_Luxembourg_2015',
  scale: 30, // Use a scale suitable for Landsat data
  region: roi,
  crs: 'EPSG:2169', // LUREF_Transverse_Mercator projection
  maxPixels: 1e13
});

